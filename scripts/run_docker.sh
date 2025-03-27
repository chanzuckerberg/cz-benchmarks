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
DEVELOPMENT_CODE_PATH=$(pwd)
EXAMPLES_CODE_PATH=${DEVELOPMENT_CODE_PATH}/examples
MOUNT_FRAMEWORK_CODE=true # true or false -- whether to mount the czbenchmarks code

# Container related settings
CZBENCH_CONTAINER_URI= # Add custom container:tag here or leave empty to use AWS container
EVAL_CMD="bash" # "bash" or "python3 -u /app/examples/example_interactive.py"
RUN_AS_ROOT=false # false or true

################################################################################
# Function definitions
# TODO -- some functions could be moved to a file and shared with other scripts

get_available_models() {    
    # Get valid models from czbenchmarks.models.utils
    local python_script="from czbenchmarks.models.utils import list_available_models; print(' '.join(list_available_models()).lower())"
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

initialize_variables() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -m|--model-name)
                MODEL_NAME=$(echo "$2" | tr '[:upper:]' '[:lower:]') # Force lowercase
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

    # Validate that required variables are set
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
    MODEL_WEIGHTS_CACHE_PATH="${MODEL_WEIGHTS_CACHE_PATH}/czbenchmarks-${MODEL_NAME}"
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

get_aws_docker_image_uri() {
    # Requires $MODEL_NAME. Must be run after initialize_variables
    if [ -z "$MODEL_NAME" ]; then
        echo -e "${RED_BOLD}MODEL_NAME is required but not set${RESET}"
        exit 1
    else
        local model_name_upper=$(echo "$MODEL_NAME" | tr '[:lower:]' '[:upper:]')
    fi

    # Get model image URI from models.yaml
    MODEL_CONFIG_PATH="conf/models.yaml"
    local python_script="import yaml; print(yaml.safe_load(open('${MODEL_CONFIG_PATH}'))['models']['${model_name_upper}']['model_image_uri'])"
    CZBENCH_CONTAINER_URI=$(python3 -c "${python_script}")

    if [ -z "$CZBENCH_CONTAINER_URI" ]; then
        echo -e "${RED_BOLD}Model ${model_name_upper} not found in ${MODEL_CONFIG_PATH}${RESET}"
        exit 1
    fi

    AWS_ECR_URI=$(echo $CZBENCH_CONTAINER_URI | cut -d/ -f1)
    AWS_REGION=$(echo $AWS_ECR_URI | cut -d. -f4)

    for var in CZBENCH_CONTAINER_URI AWS_ECR_URI AWS_REGION; do
        if [ -z "${!var}" ]; then
            echo -e "${RED_BOLD}$var is required but not set${RESET}"
            exit 1
        fi
    done

    # Login to AWS ECR and pull image
    # Alternative requires aws cli: aws ecr get-login-password --region ${AWS_REGION}
    local python_script="import boto3; print(boto3.client('ecr', region_name='${AWS_REGION}').\
    get_authorization_token()['authorizationData'][0]['authorizationToken'])"
    python3 -c "${python_script}" | base64 -d | cut -d: -f2 | \
        docker login --username AWS --password-stdin ${AWS_ECR_URI}

    docker pull ${CZBENCH_CONTAINER_URI}
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
    if [ "${USE_AWS_ECR}" = true ]; then
        echo -e "   $(printf "%-${COLUMN_WIDTH}s" "AWS_ECR_URI:") ${AWS_ECR_URI}${RESET}"
        echo -e "   $(printf "%-${COLUMN_WIDTH}s" "AWS Region:") ${AWS_REGION}${RESET}"
        echo -e "   $(printf "%-${COLUMN_WIDTH}s" "Image name:") ${CZBENCH_CONTAINER_URI}${RESET}"
    else
        echo -e "   $(printf "%-${COLUMN_WIDTH}s" "Custom image provided:") ${CZBENCH_CONTAINER_URI}${RESET}"
    fi
    echo -e "   $(printf "%-${COLUMN_WIDTH}s" "Container name:") ${CZBENCH_CONTAINER_NAME}${RESET}"

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
    for var in DATASETS_CACHE_PATH MODEL_WEIGHTS_CACHE_PATH; do
        validate_directory "${!var}" "$var"
        echo -e "   $(printf "%-${COLUMN_WIDTH}s" "${var}:") ${!var}${RESET}"
    done

    # Show user mode information
    echo ""
    echo -e "   ${GREEN_BOLD}User mode:${RESET}"
    if [ "${RUN_AS_ROOT}" = "true" ]; then
        echo -e "   Container will run as root${RESET}"
    else
        echo -e "   Container will run as current user (${USER})${RESET}"
    fi
}

build_docker_command() {
    # Build docker run command progressively
    DOCKER_CMD="docker run --rm -it \\
    --ipc=host \\
    --net=host \\
    --gpus all \\
    --shm-size=4g \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    --env TMPDIR=/tmp \\
    --env SHELL=bash \\"

    # User-specific settings if not running as root, NOTE: untested on WSL
    if [ "${RUN_AS_ROOT,,}" != "true" ]; then # Force lowercase comparison
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
        DOCKER_CMD="${DOCKER_CMD}
    --volume ${DEVELOPMENT_CODE_PATH}/docker/${MODEL_NAME}:${MODEL_CODE_PATH_DOCKER}:rw \\
    --volume ${DEVELOPMENT_CODE_PATH}:${BENCHMARK_CODE_PATH_DOCKER}:rw \\"
    fi

    # Add mount points -- examples directory must be mounted after framework code (above)
    DOCKER_CMD="${DOCKER_CMD}
    --volume ${DATASETS_CACHE_PATH}:${RAW_INPUT_DIR_PATH_DOCKER}:rw \\
    --volume ${MODEL_WEIGHTS_CACHE_PATH}:${MODEL_WEIGHTS_PATH_DOCKER}:rw \\
    --volume ${DEVELOPMENT_CODE_PATH}/examples:${EXAMPLES_CODE_PATH_DOCKER}/examples:rw \\
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
if [ -z "${CZBENCH_CONTAINER_URI}" ]; then
    get_aws_docker_image_uri
    USE_AWS_ECR=true
else
    USE_AWS_ECR=false
fi
CZBENCH_CONTAINER_NAME=$(basename ${CZBENCH_CONTAINER_URI} | tr ':' '-')
validate_variables

# Ensure docker container is updated
echo ""
echo -e "${BLUE_BOLD}########## $(printf "%-${COLUMN_WIDTH}s" "EXECUTING WORKFLOW") ##########${RESET}"

# Create and execute docker command
build_docker_command
print_docker_command
eval ${DOCKER_CMD}
