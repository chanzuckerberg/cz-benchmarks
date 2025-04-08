#!/bin/bash
#
# Interactive docker container launch script for cz-benchmarks
# See the documentation section "Running a Docker Container in 
# Interactive Mode" for detailed usage instructions

################################################################################
# User defined information

# Mount paths -- could also source from czbenchmarks.constants.py
DATASETS_CACHE_PATH=${HOME}/.cz-benchmarks/datasets
MODEL_WEIGHTS_CACHE_PATH=${HOME}/.cz-benchmarks/weights
EXAMPLES_CODE_PATH=$(pwd)/examples
DEVELOPMENT_CODE_PATH=$(pwd)
MOUNT_FRAMEWORK_CODE=true # true or false -- whether to mount the czbenchmarks code

# Container related settings
BUILD_DEV_CONTAINER=true # true or false -- true to build locally, false to pull public image
EVAL_CMD="bash" # "bash", "/opt/conda/envs/uce/bin/python -u /app/examples/example_interactive.py" for uce
# or "python3 -u /app/examples/example_interactive.py" for all other models
RUN_AS_ROOT=false # false or true

################################################################################
# Function definitions
# TODO -- some functions could be moved to a file and shared with other scripts

get_available_models() {    
    # Get valid models from czbenchmarks.models.utils
    local python_script="from czbenchmarks.models.utils import list_available_models; print(' '.join(list_available_models()).upper())"
    AVAILABLE_MODELS=($(python3 -c "${python_script}"))
    
    # Format the models as a comma-separated string for display
    AVAILABLE_MODELS_STR=$(printf ", %s" "${AVAILABLE_MODELS[@]}")
    AVAILABLE_MODELS_STR=${AVAILABLE_MODELS_STR:2}
}

print_usage() {
    echo -e "${MAGENTA_BOLD}Usage: $0 [OPTIONS]${RESET}"
    echo -e "${BOLD}Options:${RESET}"
    echo -e "  ${BOLD}-m, --model-name NAME${RESET}     Required. Set the model name, one of:"
    echo -e "  ${BOLD}${RESET}                             ( ${AVAILABLE_MODELS_STR} )"
    echo -e "  ${BOLD}${RESET}                             Model names are case-insensitive."
    echo -e "  ${BOLD}-h, --help${RESET}                Show this help message and exit."
}

validate_directory() {
    local path=$1
    local var_name=$2

    if [ -z "${path}" ]; then
        echo -e "${RED_BOLD}Error: ${var_name} is required but not set${RESET}"
        exit 1
    fi

    if [ ! -d "${path}" ]; then
        echo -e "${RED_BOLD}Error: Directory for ${var_name} does not exist: ${path}${RESET}"
        exit 1
    fi
}

variable_is_set() {
    local var_name=$1
    local var_value=${!var_name}

    if [ -z "${var_value}" ]; then
        echo -e "${RED_BOLD}Error: ${var_name} is required but not set${RESET}"
        exit 1
    fi
}

initialize_variables() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -m|--model-name)
                MODEL_NAME=$(echo "$2" | tr '[:lower:]' '[:upper:]') # Force uppercase
                shift 2
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                echo -e "${RED_BOLD}Unknown option: $1${RESET}"
                print_usage
                exit 1
                ;;
        esac
    done

    # Validate that required variables are set by flags
    if [ -z "${MODEL_NAME}" ]; then
        echo -e "${RED_BOLD}MODEL_NAME is required but not set${RESET}"
        print_usage
        exit 1
    fi
    
    # Validate that MODEL_NAME is in the list of valid models
    local is_valid=false
    for valid_model in "${AVAILABLE_MODELS[@]}"; do
        if [ "${MODEL_NAME}" = "${valid_model}" ]; then
            is_valid=true
            break
        fi
    done
    
    if [ "${is_valid}" = false ]; then # Remove leading ", "
        echo -e "${RED_BOLD}MODEL_NAME must be one of: ( ${AVAILABLE_MODELS_STR} )${RESET}"
        print_usage
        exit 1
    fi
    
    # Updates to variables which require model name and create directories if they don't exist
    local model_name_lower=$(echo "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')
    MODEL_WEIGHTS_CACHE_PATH="${MODEL_WEIGHTS_CACHE_PATH}/czbenchmarks-${model_name_lower}"
    mkdir -p ${MODEL_WEIGHTS_CACHE_PATH}

    # Docker paths -- should not be changed
    RAW_INPUT_DIR_PATH_DOCKER=/raw
    MODEL_WEIGHTS_PATH_DOCKER=/weights
    EXAMPLES_CODE_PATH_DOCKER=/app
    MODEL_CODE_PATH_DOCKER=/app
    BENCHMARK_CODE_PATH_DOCKER=/app/package # Squash existing container fw code when mounting local code

    # # Alternatively, Docker paths can also be loaded from czbenchmarks.constants.py to ensure consistency
    # PYTHON_SCRIPT="from czbenchmarks.constants import RAW_INPUT_DIR_PATH_DOCKER, MODEL_WEIGHTS_PATH_DOCKER; 
    # print(f'RAW_INPUT_DIR_PATH_DOCKER={RAW_INPUT_DIR_PATH_DOCKER}; MODEL_WEIGHTS_PATH_DOCKER={MODEL_WEIGHTS_PATH_DOCKER}')"
    # eval "$(python3 -c "${PYTHON_SCRIPT}")"
}

get_docker_image() {
    # Requires $MODEL_NAME. Must be run after initialize_variables
    variable_is_set MODEL_NAME

    echo ""
    echo -e "${MAGENTA_BOLD}########## $(printf "%-${COLUMN_WIDTH}s" "GETTING DOCKER IMAGE") ##########${RESET}"

    # Get model image URI from models.yaml
    local model_config_path="src/czbenchmarks/conf/models.yaml"
    local python_script="import yaml; print(yaml.safe_load(open('${model_config_path}'))['models']['${MODEL_NAME}']['model_image_uri'])"
    CZBENCH_CONTAINER_URI=$(python3 -c "${python_script}")
    variable_is_set CZBENCH_CONTAINER_URI

    if [ "${BUILD_DEV_CONTAINER}" = false ]; then
        echo -e "   ${MAGENTA_BOLD}Pulling image ${CZBENCH_CONTAINER_URI}${RESET}"
        docker pull ${CZBENCH_CONTAINER_URI}

    else
        local model_name_lower=$(echo "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')
        CZBENCH_CONTAINER_URI=cz-benchmarks-models:${model_name_lower}

        echo ""
        echo -e "   ${MAGENTA_BOLD}Building image ${CZBENCH_CONTAINER_URI}${RESET}"
        
        make ${model_name_lower}
    fi

    CZBENCH_CONTAINER_NAME=$(basename ${CZBENCH_CONTAINER_URI} | tr ':' '-')
    variable_is_set CZBENCH_CONTAINER_NAME
}

validate_variables() {
    echo ""
    echo -e "${GREEN_BOLD}########## $(printf "%-${COLUMN_WIDTH}s" "INITIALIZED VARIABLES") ##########${RESET}"
    echo ""
    echo -e "   ${GREEN_BOLD}Required flags:${RESET}"
    if [ ! -z "${MODEL_NAME}" ]; then
        echo -e "   $(printf "%-${COLUMN_WIDTH}s" "MODEL_NAME:") ${MODEL_NAME}${RESET}"
    else
        echo -e "${RED_BOLD}MODEL_NAME is required but not set${RESET}"
        print_usage
        exit 1
    fi

    # Show image information
    echo ""
    echo -e "   ${GREEN_BOLD}Docker setup:${RESET}"
    echo -e "   $(printf "%-${COLUMN_WIDTH}s" "Image name:") ${CZBENCH_CONTAINER_URI}${RESET}"
    echo -e "   $(printf "%-${COLUMN_WIDTH}s" "Container name:") ${CZBENCH_CONTAINER_NAME}${RESET}"
    echo -e "   $(printf "%-${COLUMN_WIDTH}s" "Docker command:") ${EVAL_CMD}${RESET}"

    if [ "${RUN_AS_ROOT}" = true ]; then
        local container_user="root"
    else
        local container_user=${USER}
    fi
    echo -e "   $(printf "%-${COLUMN_WIDTH}s" "Container user:") ${container_user}${RESET}"

    RAW_INPUT_DIR_PATH_DOCKER=/raw
    MODEL_WEIGHTS_PATH_DOCKER=/weights
    EXAMPLES_CODE_PATH_DOCKER=/app
    MODEL_CODE_PATH_DOCKER=/app
    BENCHMARK_CODE_PATH_DOCKER=/app/package # Squash existing container fw code when mounting local code

    # Show Docker paths
    local docker_paths=(RAW_INPUT_DIR_PATH_DOCKER MODEL_WEIGHTS_PATH_DOCKER MODEL_CODE_PATH_DOCKER EXAMPLES_CODE_PATH_DOCKER)
    if [ "${MOUNT_FRAMEWORK_CODE}" = true ]; then
        docker_paths+=(BENCHMARK_CODE_PATH_DOCKER)
    fi
    echo ""
    echo -e "   ${GREEN_BOLD}Docker paths:${RESET}"
    for var in "${docker_paths[@]}"; do
        echo -e "   $(printf "%-${COLUMN_WIDTH}s" "${var}:") ${!var}${RESET}"
    done

    # Examples directory
    validate_directory "${EXAMPLES_CODE_PATH}" "EXAMPLES_CODE_PATH"
    echo ""
    echo -e "   ${GREEN_BOLD}Examples path:${RESET}"
    echo -e "   $(printf "%-${COLUMN_WIDTH}s" "EXAMPLES_CODE_PATH:") ${EXAMPLES_CODE_PATH}${RESET}"
    echo -e "   EXAMPLES_CODE_PATH will be mounted in container at ${EXAMPLES_CODE_PATH_DOCKER}${RESET}"

    # Development mode
    if [ "${MOUNT_FRAMEWORK_CODE}" = true ]; then
        validate_directory "${DEVELOPMENT_CODE_PATH}" "DEVELOPMENT_CODE_PATH"
        echo ""
        echo -e "   ${GREEN_BOLD}Development paths:${RESET}"
        echo -e "   $(printf "%-${COLUMN_WIDTH}s" "DEVELOPMENT_CODE_PATH:") ${DEVELOPMENT_CODE_PATH}${RESET}"
        echo -e "   DEVELOPMENT_CODE_PATH will be mounted in container at ${BENCHMARK_CODE_PATH_DOCKER}${RESET}"
    fi
    
    # Validate required paths and show sources
    echo ""
    echo -e "   ${GREEN_BOLD}Cache paths:${RESET}"
    local paths=(DATASETS_CACHE_PATH MODEL_WEIGHTS_CACHE_PATH)
    local docker_paths=(RAW_INPUT_DIR_PATH_DOCKER MODEL_WEIGHTS_PATH_DOCKER)
    for i in "${!paths[@]}"; do
        validate_directory "${!paths[$i]}" "${paths[$i]}"
        echo -e "   $(printf "%-${COLUMN_WIDTH}s" "${paths[$i]}:") ${!paths[$i]}${RESET}"
        echo -e "   ${paths[$i]} will be mounted in container at ${!docker_paths[$i]}${RESET}"
    done
}

create_docker_run_command() {
    # Build docker run command progressively
    DOCKER_CMD="docker run --rm -it \\
    --ipc=host \\
    --net=host \\
    --gpus all \\
    --shm-size=4g \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    --env TMPDIR=/tmp \\
    --env NUMBA_CACHE_DIR=/tmp \\
    --env MPLCONFIGDIR=/tmp \\
    --env SHELL=bash \\"

    # User-specific settings if not running as root, NOTE: untested on WSL
    if [ "${RUN_AS_ROOT}" = false ]; then # Force lowercase comparison
        DOCKER_CMD="${DOCKER_CMD}
    --volume /etc/passwd:/etc/passwd:ro \\
    --volume /etc/group:/etc/group:ro \\
    --volume /etc/shadow:/etc/shadow:ro \\
    --user $(id -u):$(id -g) \\
    --volume ${HOME}/.ssh:${HOME}/.ssh:ro \\"
    fi

    # Mounts for development of cz-benchmarks framework
    # NOTE: do not change order, cz-benchmarks fw mounted last to prevent squashing
    if [ "${MOUNT_FRAMEWORK_CODE}" = true ]; then
        local model_name_lower=$(echo "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')
        local model_files=(config.yaml)

        if [ "${model_name_lower}" = "uce" ]; then
            model_files+=(uce_model.py)
        else
            model_files+=(model.py)
        fi

        if [ "${model_name_lower}" = "scvi" ]; then
            model_files+=(utils.py)
        fi

        for file in "${model_files[@]}"; do
            DOCKER_CMD="${DOCKER_CMD}
    --volume ${DEVELOPMENT_CODE_PATH}/docker/${model_name_lower}/${file}:${MODEL_CODE_PATH_DOCKER}/${file}:rw \\"
        done

        DOCKER_CMD="${DOCKER_CMD}
    --volume ${DEVELOPMENT_CODE_PATH}:${BENCHMARK_CODE_PATH_DOCKER}:rw \\"  
    fi

    # Add mount points -- examples directory must be mounted after framework code (above)
    DOCKER_CMD="${DOCKER_CMD}
    --volume ${DATASETS_CACHE_PATH}:${RAW_INPUT_DIR_PATH_DOCKER}:rw \\
    --volume ${MODEL_WEIGHTS_CACHE_PATH}:${MODEL_WEIGHTS_PATH_DOCKER}:rw \\
    --volume ${DEVELOPMENT_CODE_PATH}/examples:${EXAMPLES_CODE_PATH_DOCKER}/examples:rw \\
    --env DATASETS_CACHE_PATH=${RAW_INPUT_DIR_PATH_DOCKER} \\
    --env MODEL_WEIGHTS_CACHE_PATH=${MODEL_WEIGHTS_PATH_DOCKER} \\
    --env PYTHONPATH=${MODEL_CODE_PATH_DOCKER} \\"

    # Add final options
    DOCKER_CMD="${DOCKER_CMD}
    --env HOME=${MODEL_CODE_PATH_DOCKER} \\
    --workdir ${MODEL_CODE_PATH_DOCKER} \\
    --env MODEL_NAME=${MODEL_NAME} \\
    --name ${CZBENCH_CONTAINER_NAME} \\"

    # Add entrypoint command
    if [ "${EVAL_CMD}" = 'bash' ]; then
        DOCKER_CMD="${DOCKER_CMD}
    --entrypoint bash \\
    ${CZBENCH_CONTAINER_URI}"
    else
        DOCKER_CMD="${DOCKER_CMD}
    --entrypoint bash \\
    ${CZBENCH_CONTAINER_URI} \\
    -c \"${EVAL_CMD}\""
    fi
}

print_docker_command() {
    echo ""
    echo -e "   ${BLUE_BOLD}Executing docker command${RESET}"
    echo "${DOCKER_CMD}"
    echo ""

    # Before execution, remove extra line continuations that were for printing
    DOCKER_CMD=$(echo "${DOCKER_CMD}" | tr -d '\\')
}

################################################################################
# Main script execution starts here

# Print formatting
COLUMN_WIDTH=30
GREEN="\033[32m"
RED="\033[31m"
BLUE="\033[34m"
MAGENTA="\033[35m"
BOLD="\033[1m"
UNBOLD="\033[22m"
GREEN_BOLD="\033[32;1m"
RED_BOLD="\033[31;1m"
BLUE_BOLD="\033[34;1m"
MAGENTA_BOLD="\033[35;1m"
RESET="\033[0m"

# Check if script is run from correct directory
if [ ! "$(ls | grep -c scripts)" -eq 1 ]; then
    echo ""
    echo -e "${RED_BOLD}Run this script from root directory. Usage: bash scripts/run_docker.sh -m MODEL_NAME${RESET}"
    echo ""
    print_usage
    exit 1
fi

# Setup variables
get_available_models
initialize_variables "$@"
get_docker_image
validate_variables

# Ensure docker container is updated
echo ""
echo -e "${BLUE_BOLD}########## $(printf "%-${COLUMN_WIDTH}s" "EXECUTING WORKFLOW") ##########${RESET}"

# Create and execute docker command
create_docker_run_command
print_docker_command
eval ${DOCKER_CMD}
