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

void bli_gemm_blk_var3f( obj_t*  a,
                         obj_t*  b,
                         obj_t*  c,
                         cntx_t* cntx,
                         gemm_t* cntl,
                         gemm_thrinfo_t* thread )
{
    obj_t  a1_pack_s, b1_pack_s;

    obj_t  a1, b1;
    obj_t* a1_pack = NULL;
    obj_t* b1_pack = NULL;

	dim_t  b_alg;
	dim_t  k_trans;

    // Initialize pack objects for A and B that are passed into packm_init().
    if( thread_am_ichief( thread ) ){
        bli_obj_init_pack( &a1_pack_s );
        bli_obj_init_pack( &b1_pack_s );
    }
    a1_pack = thread_ibroadcast( thread, &a1_pack_s );
    b1_pack = thread_ibroadcast( thread, &b1_pack_s );

    cntx->b_packs[ omp_get_thread_num() ] = b1_pack;
    

    //Setup horses
    horse_t horse_s[thread->n_way];
    horse_t *horses;
    if( thread_am_ochief( thread ) ){
        for( int i = 0; i < thread->n_way; i++ ){
            setup_horse( &horse_s[i], i );
        }
    }
    horses = thread_obroadcast( thread, &horse_s[0] );


	// Query dimension in partitioning direction.
	k_trans = bli_obj_width_after_trans( *a );

    dim_t my_start, my_end;
    bli_get_range_l2r( thread, a,
                       bli_cntx_get_bmult( cntl_bszid( cntl ), cntx ),
                       &my_start, &my_end );
    
    //printf("%d\t%d\t%d\t%d\n", k_trans, my_start, my_end, thread->work_id);
	// Partition along the k dimension.
	for ( dim_t i = my_start; i < my_end; i += b_alg )
	{
		// Determine the current algorithmic blocksize.
		// NOTE: We call a gemm/hemm/symm-specific function to determine
		// the kc blocksize so that we can implement the "nudging" of kc
		// to be a multiple of mr or nr, as needed.
		b_alg = bli_gemm_determine_kc_f( i, my_end, a, b,
		                                 cntl_bszid( cntl ), cntx );

		// Acquire partitions for A1 and B1.
		bli_acquire_mpart_l2r( BLIS_SUBPART1,
		                       i, b_alg, a, &a1 );
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       i, b_alg, b, &b1 );

		// Initialize objects for packing A1 and B1.
        if( thread_am_ichief( thread ) ) {
            bli_packm_init( &a1, a1_pack,
                            cntx, cntl_sub_packm_a( cntl ) );
            bli_packm_init( &b1, b1_pack,
                            cntx, cntl_sub_packm_b( cntl ) );
        }
        thread_ibarrier( thread );

		// Pack A1 (if instructed).
		bli_packm_int( &a1, a1_pack,
		               cntx, cntl_sub_packm_a( cntl ),
                       gemm_thread_sub_ipackm( thread ) );

		// Pack B1 (if instructed).
        char* blah = getenv("BLIS_OVERLAP");
        if( blah[0] != 1 ){
		bli_packm_int( &b1, b1_pack,
		               cntx, cntl_sub_packm_b( cntl ),
                       gemm_thread_sub_ipackm( thread ) );
        }

		// Perform gemm subproblem.
       /* mount_horse( &horses[0] );
        printf("work id %d\n starting\n", thread->work_id );
		bli_gemm_int( &BLIS_ONE,
		              a1_pack,
		              b1_pack,
		              &BLIS_ONE,
		              c,
		              cntx,
		              cntl_sub_gemm( cntl ),
                      gemm_thread_sub_gemm( thread) ); 
        printf("work id %d\n done\n", thread->work_id );
        unmount_horse( &horses[0] );*/

        mutex_carousel( horses, thread->n_way, thread->work_id, CAROUSEL_DIR_M,
                        (l3_int_t) bli_gemm_int, &BLIS_ONE, a1_pack, b1_pack, &BLIS_ONE, c, cntx, cntl_sub_gemm( cntl ), (thrinfo_t*) gemm_thread_sub_gemm( thread ) );

		// This variant executes multiple rank-k updates. Therefore, if the
		// internal beta scalar on matrix C is non-zero, we must use it
		// only for the first iteration (and then BLIS_ONE for all others).
		// And since c_pack is a local obj_t, we can simply overwrite the
		// internal beta scalar with BLIS_ONE once it has been used in the
		// first iteration.
        // This must happen during the carousel
		// if ( i == 0 && thread_am_ichief( thread ) ) bli_obj_scalar_reset( c_pack );

        thread_ibarrier( thread );

	}

    thread_obarrier( thread );
	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
    if( thread_am_ichief( thread ) ){
        bli_packm_release( a1_pack, cntl_sub_packm_a( cntl ) );
        bli_packm_release( b1_pack, cntl_sub_packm_b( cntl ) );
    }
}
