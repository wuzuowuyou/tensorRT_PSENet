/*---------------------------configuration parameter---------------------------------*/
//#define SERIALIZE
#define INFER


// SERIALIZE 序列化的时候需要指定path_wts  path_save_engin
std::string path_wts = "../attach/psenet2trt_wts/psenet0419.wts";
std::string path_save_engin = "./psenet_0419.engine";


//INFER 推理的时候需要指定path_read_engin
std::string path_read_engin = "./psenet_0419.engine";

//图片文件夹名
const char * dir_img = "../attach/test_data/";
bool b_show = 1;

bool b_test_save_dir = 0;//为了测试precision, recall, hmean,保存txt
std::string save_path = "../attach/";

int long_size = 1280;
int kernel_start_id = 0;
int kernel_num = 3;

#define FP32  // FP32  FP16   comment out this if want to use FP32
#define DEVICE 0  // GPU id
static const int MAX_INPUT_SIZE = 1280; // 32x
static const int MIN_INPUT_SIZE = 320;
static const int OPT_INPUT_W = 1280;
static const int OPT_INPUT_H = 640;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "out";
/*---------------------------configuration parameter---------------------------------*/