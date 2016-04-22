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
#include "carousel.h"

void bli_gemm_blk_var1f( obj_t*  a,
                         obj_t*  b,
                         obj_t*  c,
                         cntx_t* cntx,
                         gemm_t* cntl,
                         gemm_thrinfo_t* thread )
{
    //The s is for "lives on the stack"
    obj_t a1_pack_s, c1_pack_s;

    obj_t a1, c1;
    obj_t* a1_pack  = NULL;
    obj_t* c1_pack  = NULL;

	dim_t i;
	dim_t b_alg;

	// Initialize objects passed into bli_packm_init for A and C
    if( thread_am_ichief( thread ) ) {
        bli_obj_init_pack( &a1_pack_s );
        bli_obj_init_pack( &c1_pack_s );
    }
    a1_pack = thread_ibroadcast( thread, &a1_pack_s );
    c1_pack = thread_ibroadcast( thread, &c1_pack_s );

    //Setup camels 
    camel_t camels_s[thread->n_way];
    camel_t *camels;
    if( thread_am_ochief( thread ) ){
        for( int i = 0; i < thread->n_way; i++ ){
            setup_camel( &camels_s[i] );
        }
    }
    camels = thread_obroadcast( thread, &camels_s[0] );

    dim_t my_start, my_end;
    bli_get_range_t2b( thread, a,
                       bli_cntx_get_bmult( cntl_bszid( cntl ), cntx ),
                       &my_start, &my_end );

	// Partition along the m dimension.
	for ( i = my_start; i < my_end; i += b_alg )
	{
		// Determine the current algorithmic blocksize.
		// NOTE: Use of a (for execution datatype) is intentional!
		// This causes the right blocksize to be used if c and a are
		// complex and b is real.
		b_alg = bli_determine_blocksize_f( i, my_end, a,
		                                   cntl_bszid( cntl ), cntx );

		// Acquire partitions for A1 and C1.
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       i, b_alg, a, &a1 );
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       i, b_alg, c, &c1 );
		
        // Initialize objects for packing A1 and C1.
        if( thread_am_ichief( thread ) ) {
            bli_packm_init( &a1, a1_pack,
                            cntx, cntl_sub_packm_a( cntl ) );
            bli_packm_init( &c1, c1_pack,
                            cntx, cntl_sub_packm_c( cntl ) );
        }
        thread_ibarrier( thread );

		// Pack A1 (if instructed).
		bli_packm_int( &a1, a1_pack,
		               cntx, cntl_sub_packm_a( cntl ),
                       gemm_thread_sub_ipackm( thread ) );

		// Pack C1 (if instructed).
		bli_packm_int( &c1, c1_pack,
		               cntx, cntl_sub_packm_c( cntl ),
                       gemm_thread_sub_ipackm( thread ) );

		// Perform gemm subproblem.
        char* blah = getenv("BLIS_OVERLAP");
        if( blah != NULL && blah[0] != 1 ) {
            preparation_carousel( camels, thread->n_way, thread->work_id, CAROUSEL_DIR_N,
                                 (l3_int_t) bli_gemm_ker_var2_overlap,
                                 (l3_int_t) bli_gemm_ker_var2, &BLIS_ONE, a1_pack, b, &BLIS_ONE, c1_pack, cntx, cntl_sub_gemm( cntl ), 
                                 (thrinfo_t*) gemm_thread_sub_gemm( thread ) );
        }
        else{ 
		    bli_gemm_int( &BLIS_ONE,
		              a1_pack,
		              b,
		              &BLIS_ONE,
		              c1_pack,
		              cntx,
		              cntl_sub_gemm( cntl ),
                      gemm_thread_sub_gemm( thread ) );
        }

        thread_ibarrier( thread );

		// Unpack C1 (if C1 was packed).
        // Currently must be done by 1 thread
        bli_unpackm_int( c1_pack, &c1,
                         cntx, cntl_sub_unpackm_c( cntl ),
                         gemm_thread_sub_ipackm( thread ) );
	}

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
    thread_obarrier( thread );
    if( thread_am_ichief( thread ) ){
        bli_packm_release( a1_pack, cntl_sub_packm_a( cntl ) );
        bli_packm_release( c1_pack, cntl_sub_packm_c( cntl ) );
    }
}

