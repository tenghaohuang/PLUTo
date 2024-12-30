class Config:
    """
    Environemnt Variables
    """
    OPENAI_API_KEY=""

    DESCRIPTION_PATH="data/"
    q4opt_SAVE_PATH='queries4optimization/'
    TRAIN_DATA_PATH="toolbench_v2/train"
    DEV_DATA_PATH="toolbench_v2/dev"
    TEST_DATA_PATH="toolbench_v2/test"
    DESCRIPTION_INDEX_PATH="index/"

    QUERY_ENCODER="facebook-dpr-question_encoder-single-nq-base"
    PASSAGE_ENCODER="facebook-dpr-ctx_encoder-single-nq-base"
    # QUERY_ENCODER="intfloat/e5-large-v2"
    # PASSAGE_ENCODER="intfloat/e5-large-v2"
    # DESCRIPTION_PATH="descriptions/original_description.p"
    DESCRIPTION_PATH="data/tool_dict.p"
    # DESCRIPTION_PATH="data/3300_optimized_description.p"
    # TRAIN_DATA_PATH="data/multi_tool_query_gts_data.p"
    TRAIN_DATA_PATH="toolbench_v2/toolbench_train_curated_18k.p"
    DATA_LOAD_LIMIT=10
    DEV_DATA_PATH="toolbench_v2/toolbench_v2_dev.p"
    TEST_DATA_PATH="toolbench_v2/toolbench_v2_test.p"
    DESCRIPTION_INDEX_PATH="index/description_index.index"
    USECASE_INDEX_PATH="index/usecase_index.index"
    
    """
    Model Variables
    """
    MODEL_NAME="gpt-3.5-turbo"
    #MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
    DEFAULT_QUERY_ENCODER="facebook-dpr-question_encoder-single-nq-base"
    DEFAULT_PASSAGE_ENCODER="facebook-dpr-ctx_encoder-single-nq-base"
    OPTIMIZE_THRESHOLD = 3

    DATA_LOAD_LIMIT=10
    LM_SCORING='False'
    
    """
    Execute Mode
    """
    OPTIMIZE='False'

    """
    Experiment Settings
    """
    #DESCRIPTION_FILE="tool_dict.p"
    DESCRIPTION_FILE="11_10_3300_optimized_description.p"

    #DPR_PATH="finetune/saved_models/11_10_optimized_finetuned_dpr/" 
    DPR_PATH="default_dpr"
    #DPR_PATH="finetune/saved_models/tool_dict_finetuned_dpr/"
    
    PROCEDURAL_REASONING='False'
    
    RETRIEVE_WITH_USECASE='True'
    
    
    

    API2Q_DICT = "toolbench_v2/api2q_curated_50.p"
    q4opt_SAVE_PATH = 'queries4optimization/'
    ERASE_OPTIMIZE_CACHE=True

