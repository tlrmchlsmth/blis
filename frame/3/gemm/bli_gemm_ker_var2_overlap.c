/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include "bli_packm_cxk.h"
#define FUNCPTR_T gemm_fp

typedef void (*FUNCPTR_T)(
                           pack_t  schema_a,
                           pack_t  schema_b,
                           dim_t   m,
                           dim_t   n,
                           dim_t   k,
                           void*   alpha,
                           void*   a_pack, inc_t cs_a_pack, inc_t is_a_pack,
                                      dim_t pd_a_pack, inc_t ps_a_pack,
                           void*   b_orig, inc_t rs_b_orig, inc_t cs_b_orig,
                           void*   b_pack, inc_t rs_b_pack, inc_t is_b_pack,
                                      dim_t pd_b_pack, inc_t ps_b_pack,
                           bool_t  pack_b,
                           void*   beta,
                           void*   c, inc_t rs_c, inc_t cs_c,
                           void*   gemm_ukr,
                           gemm_thrinfo_t* thread
                         );

static FUNCPTR_T GENARRAY(ftypes,gemm_ker_var2_overlap);


void bli_gemm_ker_var2_overlap( obj_t*  a,
                        obj_t*  b,
                        obj_t*  c,
                        cntx_t* cntx,
                        gemm_t* cntl,
                        gemm_thrinfo_t* thread )
{
    bool_t  pack_b = 1;
	num_t     dt_exec   = bli_obj_execution_datatype( *c );

	pack_t    schema_a  = bli_obj_pack_schema( *a_pack );
	pack_t    schema_b  = bli_obj_pack_schema( *b_pack );

	dim_t     m         = bli_obj_length( *c );
	dim_t     n         = bli_obj_width( *c );
	dim_t     k         = bli_obj_width( *a_pack );

	void*     buf_a_pack     = bli_obj_buffer_at_off( *a_pack );
	inc_t     cs_a_pack      = bli_obj_col_stride( *a_pack );
	inc_t     is_a_pack      = bli_obj_imag_stride( *a_pack );
	dim_t     pd_a_pack      = bli_obj_panel_dim( *a_pack );
	inc_t     ps_a_pack      = bli_obj_panel_stride( *a_pack );

    obj_t* b_orig = cntx->b_packs[ omp_get_thread_num() ];
	void*     buf_b_orig = bli_obj_buffer_at_off( *b_orig );
    inc_t     rs_b_orig  = bli_obj_row_stride( *b_orig );
    inc_t     cs_b_orig  = bli_obj_col_stride( *b_orig );

	void*     buf_b_pack = bli_obj_buffer_at_off( *b_pack );
	inc_t     rs_b_pack  = bli_obj_row_stride( *b_pack );
	inc_t     is_b_pack  = bli_obj_imag_stride( *b_pack );
	dim_t     pd_b_pack  = bli_obj_panel_dim( *b_pack );
	inc_t     ps_b_pack  = bli_obj_panel_stride( *b_pack );

	void*     buf_c     = bli_obj_buffer_at_off( *c );
	inc_t     rs_c      = bli_obj_row_stride( *c );
	inc_t     cs_c      = bli_obj_col_stride( *c );

	obj_t     scalar_a;
	obj_t     scalar_b;

	void*     buf_alpha;
	void*     buf_beta;

	FUNCPTR_T f;

	func_t*   gemm_ukrs;
	void*     gemm_ukr;


	// Detach and multiply the scalars attached to A and B.
	bli_obj_scalar_detach( a_pack, &scalar_a );
	bli_obj_scalar_detach( b_pack, &scalar_b );
	bli_mulsc( &scalar_a, &scalar_b );

	// Grab the addresses of the internal scalar buffers for the scalar
	// merged above and the scalar attached to C.
	buf_alpha = bli_obj_internal_scalar_buffer( scalar_b );
	buf_beta  = bli_obj_internal_scalar_buffer( *c );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_exec];

	// Extract from the control tree node the func_t object containing
	// the gemm micro-kernel function addresses, and then query the
	// function address corresponding to the current datatype.
	gemm_ukrs = cntl_gemm_ukrs( cntl );
	gemm_ukr  = bli_func_obj_query( dt_exec, gemm_ukrs );

	// Invoke the function.
	f( schema_a,
	   schema_b,
	   m,
	   n,
	   k,
	   buf_alpha,
	   buf_a_pack, cs_a_pack, is_a_pack,
	          pd_a_pack, ps_a_pack,
       buf_b_orig, cs_b_orig, rs_b_orig,
	   buf_b_pack, rs_b_pack, is_b_pack,
	          pd_b_pack, ps_b_pack,
       pack_b,
	   buf_beta,
	   buf_c, rs_c, cs_c,
	   gemm_ukr,
	   thread );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, ukrtype ) \
\
void PASTEMAC(ch,varname)( \
                           pack_t  schema_a, \
                           pack_t  schema_b, \
                           dim_t   m, \
                           dim_t   n, \
                           dim_t   k, \
                           void*   alpha, \
                           void*   a_pack, inc_t cs_a_pack, inc_t is_a_pack, \
                                      dim_t pd_a_pack, inc_t ps_a_pack, \
                           void*   b_orig, inc_t rs_b_orig, inc_t cs_b_orig, \
                           void*   b_pack, inc_t rs_b_pack, inc_t is_b_pack, \
                                      dim_t pd_b_pack, inc_t ps_b_pack, \
                           bool_t  pack_b, \
                           void*   beta, \
                           void*   c, inc_t rs_c, inc_t cs_c, \
                           void*   gemm_ukr,  \
                           gemm_thrinfo_t* thread \
                         ) \
{ \
	/* Cast the micro-kernel address to its function pointer type. */ \
	PASTECH(ch,ukrtype) gemm_ukr_cast = gemm_ukr; \
\
	/* Temporary C buffer for edge cases. */ \
	ctype           ct[ PASTEMAC(ch,maxmr) * \
	                    PASTEMAC(ch,maxnr) ] \
	                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const inc_t     rs_ct      = 1; \
	const inc_t     cs_ct      = PASTEMAC(ch,maxmr); \
\
	/* Alias some constants to simpler names. */ \
	const dim_t     MR         = pd_a_pack; \
	const dim_t     NR         = pd_b_pack; \
	/*const dim_t     PACKMR     = cs_a;*/ \
	/*const dim_t     PACKNR     = rs_b;*/ \
\
	ctype* restrict zero       = PASTEMAC(ch,0); \
	ctype* restrict a_pack_cast     = a_pack; \
	ctype* restrict b_orig_cast = b_orig; \
	ctype* restrict b_pack_cast = b_pack; \
	ctype* restrict c_cast     = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict beta_cast  = beta; \
	ctype* restrict b_orig_1; \
	ctype* restrict b_pack_1; \
	ctype* restrict c1; \
\
	dim_t           m_iter, m_left; \
	dim_t           n_iter, n_left; \
	dim_t           i, j; \
	dim_t           m_cur; \
	dim_t           n_cur; \
	inc_t           rstep_a_pack; \
	inc_t           cstep_b_orig; \
	inc_t           cstep_b_pack; \
	inc_t           rstep_c, cstep_c; \
	auxinfo_t       aux; \
\
	/*
	   Assumptions/assertions:
	     rs_a == 1
	     cs_a == PACKMR
	     pd_a == MR
	     ps_a == stride to next micro-panel of A
	     rs_b == PACKNR
	     cs_b == 1
	     pd_b == NR
	     ps_b == stride to next micro-panel of B
	     rs_c == (no assumptions)
	     cs_c == (no assumptions)
	*/ \
\
	/* If any dimension is zero, return immediately. */ \
	if ( bli_zero_dim3( m, n, k ) ) return; \
\
	/* Clear the temporary C buffer in case it has any infs or NaNs. */ \
	PASTEMAC(ch,set0s_mxn)( MR, NR, \
	                        ct, rs_ct, cs_ct ); \
\
	/* Compute number of primary and leftover components of the m and n
	   dimensions. */ \
	n_iter = n / NR; \
	n_left = n % NR; \
\
	m_iter = m / MR; \
	m_left = m % MR; \
\
	if ( n_left ) ++n_iter; \
	if ( m_left ) ++m_iter; \
\
	/* Determine some increments used to step through A, B, and C. */ \
	rstep_a_pack = ps_a_pack; \
\
	cstep_b_orig = rs_b_orig * NR; \
	cstep_b_pack = ps_b_pack; \
\
	rstep_c = rs_c * MR; \
	cstep_c = cs_c * NR; \
\
	/* Save the pack schemas of A and B to the auxinfo_t object. */ \
	bli_auxinfo_set_schema_a( schema_a, aux ); \
	bli_auxinfo_set_schema_b( schema_b, aux ); \
\
	/* Save the imaginary stride of A and B to the auxinfo_t object. */ \
	bli_auxinfo_set_is_a( is_a_pack, aux ); \
	bli_auxinfo_set_is_b( is_b_pack, aux ); \
\
	gemm_thrinfo_t* caucus = gemm_thread_sub_gemm( thread ); \
	dim_t jr_num_threads = thread_n_way( thread ); \
	dim_t jr_thread_id   = thread_work_id( thread ); \
	dim_t ir_num_threads = thread_n_way( caucus ); \
	dim_t ir_thread_id   = thread_work_id( caucus ); \
\
	/* Loop over the n dimension (NR columns at a time). */ \
	for ( j = jr_thread_id; j < n_iter; j += jr_num_threads ) \
	{ \
		ctype* restrict a_pack_1; \
		ctype* restrict c11; \
		ctype* restrict b2; \
\
		b_pack_1 = b_pack_cast + j * cstep_b_pack; \
        b_orig_1 = b_orig_cast + j * cstep_b_orig; \
		c1 = c_cast + j * cstep_c; \
\
        /* Assertion: Pack the micro-panel of B if instructed */ \
        if( pack_b ) { \
            double one = 1.0;\
            PASTEMAC(ch,packm_cxk)( \
                BLIS_NO_CONJUGATE, \
                NR, \
                k, \
                (void*) &one, \
                b_orig_1, rs_b_orig, cs_b_orig, \
                b_pack_1, pd_b_pack );\
        } \
\
\
		n_cur = ( bli_is_not_edge_f( j, n_iter, n_left ) ? NR : n_left ); \
\
		/* Initialize our next panel of B to be the current panel of B. */ \
		b2 = b_pack_1; \
\
		/* Loop over the m dimension (MR rows at a time). */ \
		for ( i = ir_thread_id; i < m_iter; i += ir_num_threads ) \
		{ \
			ctype* restrict a2; \
\
			a_pack_1 = a_pack_cast + i * rstep_a_pack; \
			c11 = c1     + i * rstep_c; \
			m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left ); \
\
			/* Compute the addresses of the next panels of A and B. */ \
			a2 = gemm_get_next_a_micropanel( caucus, a_pack_1, rstep_a_pack ); \
			if ( bli_is_last_iter( i, m_iter, ir_thread_id, ir_num_threads ) ) \
			{ \
				a2 = a_pack_cast; \
				b2 = gemm_get_next_b_micropanel( thread, b_pack_1, cstep_b_pack ); \
				if ( bli_is_last_iter( j, n_iter, jr_thread_id, jr_num_threads ) ) \
					b2 = b_pack_cast; \
			} \
\
			/* Save addresses of next panels of A and B to the auxinfo_t
			   object. */ \
			bli_auxinfo_set_next_a( a2, aux ); \
			bli_auxinfo_set_next_b( b2, aux ); \
\
			/* Handle interior and edge cases separately. */ \
			if ( m_cur == MR && n_cur == NR ) \
			{ \
				/* Invoke the gemm micro-kernel. */ \
				gemm_ukr_cast( k, \
				               alpha_cast, \
				               a_pack_1, \
				               b_pack_1, \
				               beta_cast, \
				               c11, rs_c, cs_c, \
				               &aux ); \
			} \
			else \
			{ \
				/* Invoke the gemm micro-kernel. */ \
				gemm_ukr_cast( k, \
				               alpha_cast, \
				               a_pack_1, \
				               b_pack_1, \
				               zero, \
				               ct, rs_ct, cs_ct, \
				               &aux ); \
\
				/* Scale the bottom edge of C and add the result from above. */ \
				PASTEMAC(ch,xpbys_mxn)( m_cur, n_cur, \
				                        ct,  rs_ct, cs_ct, \
				                        beta_cast, \
				                        c11, rs_c,  cs_c ); \
			} \
		} \
	} \
\
/*PASTEMAC(ch,fprintm)( stdout, "gemm_ker_var2: b1", k, NR, b1, NR, 1, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemm_ker_var2: a1", MR, k, a1, 1, MR, "%4.1f", "" );*/ \
}

INSERT_GENTFUNC_BASIC( gemm_ker_var2_overlap, gemm_ukr_t )

