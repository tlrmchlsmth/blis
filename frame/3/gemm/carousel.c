#include "blis.h"
#include "carousel.h"

#ifdef BLIS_ENABLE_OPENMP
void setup_horse( horse_t* horse )
{
    omp_init_lock( &horse->mutex );
}

void cleanup_horse( horse_t* horse )
{
    omp_destroy_lock( &horse->mutex );
}

void mount_horse( horse_t* horse )
{
    omp_set_lock( &horse->mutex );
}

void unmount_horse( horse_t* horse )
{
    omp_unset_lock( &horse->mutex );
}
#endif

#ifdef BLIS_ENABLE_PTHREADS
void setup_horse( horse_t* horse )
{
    pthread_mutex_init( &horse->mutex, NULL );
}

void cleanup_horse( horse_t* horse )
{
    pthread_mutex_destroy( &horse->mutex );
}

void mount_horse( horse_t* horse )
{
    pthread_mutex_lock( &horse->mutex );
}

void unmount_horse( horse_t* horse )
{
    pthread_mutex_unlock( &horse->mutex );
}
#endif

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
        dim_t bf = bli_cntx_get_blksz_def_dt( bli_obj_execution_datatype( *a ), BLIS_MR, cntx );
        
        for( int i = work_id; i < n_horses; i++)
        {
            dummy.work_id = i;
            bli_get_range( &dummy, m, bf, TRUE, &start, &end );
            if(end - start <= 0) break;
            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, a, &a1 );
            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, c, &c1 );
            
            if(thread_am_ochief( thread ) ){
                mount_horse( &horses[i] );
            }
            thread_obarrier( thread );
            
            subproblem( alpha, &a1, b, &BLIS_ONE, &c1, cntx, cntl, thread );

            if(thread_am_ochief( thread ) ){
                unmount_horse( &horses[i] );
            }
        }
        for( int i = 0; i < work_id; i++ )
        {
            dummy.work_id = i;
            bli_get_range( &dummy, m, bf, TRUE, &start, &end );
            if(end - start <= 0) break;
            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, a, &a1 );
            bli_acquire_mpart_t2b( BLIS_SUBPART1, start, end-start, c, &c1 );

            if(thread_am_ochief( thread ) ){
                mount_horse( &horses[i] );
            }
            thread_obarrier( thread );
            
            subproblem( alpha, &a1, b, &BLIS_ONE, &c1, cntx, cntl, thread );

            if(thread_am_ochief( thread ) ){
                unmount_horse( &horses[i] );
            }
        }
    }
    else{
        bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
    }
}