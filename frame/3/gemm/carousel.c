#include "blis.h"
#include "carousel.h"
#include "sched.h"

void setup_horse( horse_t* horse, int init_rider )
{
    horse->next_rider = init_rider;
}

void cleanup_horse( horse_t* horse )
{
}

void mount_horse( horse_t* horse, int rider_id )
{
    while( horse->next_rider != rider_id) {  }
}

void unmount_horse( horse_t* horse, int n_riders )
{
    int next_rider = horse->next_rider - 1;
    if( next_rider == -1 ) next_rider = n_riders - 1;
    horse->next_rider = next_rider;
}

void mutex_carousel( horse_t* horses, dim_t n_horses, dim_t work_id, carousel_dir_t direction,
                    l3_int_t subproblem, obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c,
                    cntx_t* cntx, gemm_t* cntl, thrinfo_t* thread )
{
    thrinfo_t dummy;
    dummy.n_way = n_horses;
    if(direction == CAROUSEL_DIR_M )
    {
        obj_t a1, c1;
        dim_t start, end;
        dim_t m = bli_obj_length_after_trans( *a );
        dim_t bf = bli_cntx_get_blksz_def_dt( bli_obj_datatype( *a ), BLIS_MR, cntx );

        for( int i = work_id; i < n_horses; i++)
        {
            dummy.work_id = i;
            bli_get_range( &dummy, m, bf, FALSE, &start, &end );

            //If there's no work, we still have to get on the horse and then back off.
            //(to prevent deadlocks)
            if(end - start <= 0) {
                if(thread_am_ochief( thread ) ){ mount_horse( &horses[i], work_id ); }
                if(thread_am_ochief( thread ) ){ unmount_horse( &horses[i], n_horses ); }
                continue;
            }

            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, a, &a1 );
            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, c, &c1 );
        
            if(thread_am_ochief( thread ) ){
                mount_horse( &horses[i], work_id );
            }
            thread_obarrier( thread );
            
            subproblem( alpha, &a1, b, &BLIS_ONE, &c1, cntx, cntl, thread );
            
            //Only scale by beta during the work_id iteration.
            if( i == work_id ){
                thread_obarrier( thread );
		        if ( thread_am_ochief( thread ) ) bli_obj_scalar_reset( c );
            }

            if(thread_am_ochief( thread ) ){
                unmount_horse( &horses[i], n_horses );
            }
        }
        for( int i = 0; i < work_id; i++ )
        {
            dummy.work_id = i;
            bli_get_range( &dummy, m, bf, FALSE, &start, &end );

            //If there's no work, we still have to get on the horse and then back off.
            //(to prevent deadlocks)
            if(end - start <= 0) {
                if(thread_am_ochief( thread ) ){ mount_horse( &horses[i], work_id ); }
                if(thread_am_ochief( thread ) ){ unmount_horse( &horses[i], n_horses ); }
                continue;
            }
            
            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, a, &a1 );
            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, c, &c1 );

            if(thread_am_ochief( thread ) ){
                mount_horse( &horses[i], work_id );
            }
            thread_obarrier( thread );
            
            subproblem( alpha, &a1, b, &BLIS_ONE, &c1, cntx, cntl, thread );

            if(thread_am_ochief( thread ) ){
                unmount_horse( &horses[i], n_horses );
            }
        }
    }
    else{
        bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
    }
}

void setup_camel( camel_t* camel )
{
    camel->saddled = 0;
}
void cleanup_camel( camel_t* camel )
{
}
void saddle_camel( camel_t* camel )
{
    camel->saddled = 1;
}
void wait_for_saddle( camel_t* camel )
{
    while( !camel->saddled ){ }
}

void preparation_carousel( camel_t* camels, dim_t n_camels, dim_t work_id, carousel_dir_t direction,
                    l3_int2_t prepare_subproblem,
                    l3_int2_t subproblem, obj_t* a, obj_t* b, obj_t* c,
                    cntx_t* cntx, gemm_t* cntl, thrinfo_t* thread )
{
    obj_t* b_orig = cntx->b_origs[omp_get_thread_num()];

    thrinfo_t dummy;
    dummy.n_way = n_camels;
    if(direction == CAROUSEL_DIR_M )
    {
        obj_t a1, c1;
        dim_t start, end;
        dim_t m = bli_obj_length_after_trans( *a );
        dim_t bf = bli_cntx_get_blksz_def_dt( bli_obj_datatype( *a ), BLIS_MR, cntx );

        //First, saddle our camel
        dummy.work_id = work_id;
        bli_get_range( &dummy, m, bf, FALSE, &start, &end );
        if( end - start <= 0 ) {
            if( thread_am_ochief( thread ) ) saddle_camel( &camels[work_id] );
        }
        else {
            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, a, &a1 );
            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, c, &c1 );

            prepare_subproblem( &a1, b, &c1, cntx, cntl, thread );

            thread_obarrier( thread );
            if( thread_am_ochief( thread ) ) saddle_camel( &camels[work_id] );
        }

        //Now use other camels too
        for( int i = work_id+1; i < n_camels; i++)
        {
            dummy.work_id = i;
            bli_get_range( &dummy, m, bf, FALSE, &start, &end );

            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, a, &a1 );
            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, c, &c1 );
        
            wait_for_saddle( &camels[i] );
            subproblem( &a1, b, &c1, cntx, cntl, thread );
        }
        for( int i = 0; i < work_id; i++ )
        {
            dummy.work_id = i;
            bli_get_range( &dummy, m, bf, FALSE, &start, &end );

            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, a, &a1 );
            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, c, &c1 );

            wait_for_saddle( &camels[i] );
            subproblem( &a1, b, &c1, cntx, cntl, thread );
        }
    }
    else if(direction == CAROUSEL_DIR_N ) {
        obj_t b1, c1, b_orig1;
        dim_t start, end;
        dim_t n = bli_obj_width_after_trans( *b );
        dim_t bf = bli_cntx_get_blksz_def_dt( bli_obj_datatype( *b ), BLIS_NR, cntx );

        //First, saddle our camel
        dummy.work_id = work_id;
        bli_get_range( &dummy, n, bf, FALSE, &start, &end );
    
        if( end - start <= 0 ) {
            if( thread_am_ochief( thread ) ){
                saddle_camel( &camels[work_id] );
            }
        }
        else {
            bli_acquire_mpart_l2r( BLIS_SUBPART1, start, end-start, b, &b1 );
            bli_acquire_mpart_l2r( BLIS_SUBPART1, start, end-start, c, &c1 );
            bli_acquire_mpart_l2r( BLIS_SUBPART1, start, end-start, b_orig, &b_orig1 );
            cntx->b_origs[omp_get_thread_num()] = &b_orig1;
            
            prepare_subproblem( a, &b1, &c1, cntx, cntl, thread );

            thread_obarrier( thread );
            if( thread_am_ochief( thread ) ) saddle_camel( &camels[work_id] );
        }
        
        //Now use other camels too
        for( int i = work_id+1; i < n_camels; i++)
        {
            dummy.work_id = i;
            bli_get_range( &dummy, n, bf, FALSE, &start, &end );

            bli_acquire_mpart_l2r( BLIS_SUBPART1, start, end-start, b, &b1 );
            bli_acquire_mpart_l2r( BLIS_SUBPART1, start, end-start, c, &c1 );
            bli_acquire_mpart_l2r( BLIS_SUBPART1, start, end-start, b_orig, &b_orig1 );
            cntx->b_origs[omp_get_thread_num()] = &b_orig1;


        
            wait_for_saddle( &camels[i] );
            subproblem( a, &b1, &c1, cntx, cntl, thread );
        }
        for( int i = 0; i < work_id; i++ )
        {
            dummy.work_id = i;
            bli_get_range( &dummy, n, bf, FALSE, &start, &end );

            bli_acquire_mpart_l2r( BLIS_SUBPART1, start, end-start, b, &b1 );
            bli_acquire_mpart_l2r( BLIS_SUBPART1, start, end-start, c, &c1 );
            bli_acquire_mpart_l2r( BLIS_SUBPART1, start, end-start, b_orig, &b_orig1 );
            cntx->b_origs[omp_get_thread_num()] = &b_orig1;
        
            wait_for_saddle( &camels[i] );
            subproblem( a, &b1, &c1, cntx, cntl, thread );
        }
    }
    else
    {
/*        obj_t b1, c1;
        dim_t start, end;
//        dim_t n = bli_obj_width_after_trans( *b );
//        dim_t bf = bli_cntx_get_blksz_def_dt( bli_obj_datatype( *b ), BLIS_NR, cntx );

        //First, saddle our camel
//        dummy.work_id = work_id;
//        bli_get_range( &dummy, n, bf, FALSE, &start, &end );

//#define NOSPLIT
#ifdef NOSPLIT
        bli_acquire_mpart_l2r( BLIS_SUBPART1, 0, 6, b, &b1 );
        bli_acquire_mpart_l2r( BLIS_SUBPART1, 0, 6, c, &c1 );
        prepare_subproblem( a, &b1, &c1, cntx, cntl, thread );
#else
        obj_t ba, bb;
        bli_acquire_mpart_l2r( BLIS_SUBPART1, 0, 3, b, &ba );
        bli_acquire_mpart_l2r( BLIS_SUBPART1, 0, 3, c, &c1 );
        prepare_subproblem( a, &ba, &c1, cntx, cntl, thread );

        bli_acquire_mpart_l2r( BLIS_SUBPART1, 3, 3, b, &bb );
        bli_acquire_mpart_l2r( BLIS_SUBPART1, 3, 3, c, &c1 );
        prepare_subproblem( a, &bb, &c1, cntx, cntl, thread );
#endif
*/
    }
    cntx->b_origs[omp_get_thread_num()] = b_orig;
}
