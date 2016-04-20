struct horse_s {
    int next_rider;
};
typedef struct horse_s horse_t;

enum carousel_dir_e
{
    CAROUSEL_DIR_M,
    CAROUSEL_DIR_N,
    CAROUSEL_DIR_K,
    CAROUSEL_DIR_MN,
};
typedef enum carousel_dir_e carousel_dir_t;

void setup_horse( horse_t* horse, int init_rider );
void cleanup_horse( horse_t* horse );
void mount_horse( horse_t* horse, int rider_id );
void unmount_horse( horse_t* horse, int num_riders );


void mutex_carousel( horse_t* horses, dim_t n_horses, dim_t work_id, carousel_dir_t direction,
                    l3_int_t subproblem, obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c,
                    cntx_t* cntx, gemm_t* cntl, thrinfo_t* thread );


