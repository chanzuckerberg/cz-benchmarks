#!/bin/bash
#
# Interactive docker container launch script for cz-benchmarks
# See the documentation section "Running a Docker Container in Interactive Mode" for detailed usage instructions

################################################################################
# User defined information

# Mount paths
DATASETS_CACHE_PATH=${HOME}/.cz-benchmarks/datasets
MODEL_WEIGHTS_CACHE_PATH=${HOME}/.cz-benchmarks/weights
DEVELOPMENT_CODE_PATH=$(pwd) # Leave blank or remove to not mount code

# Container execution settings
EVAL_CMD=bash # e.g. bash or "python3 examples/example_interactive.py"
RUN_AS_ROOT=false # false or true

################################################################################
# Function definitions
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -m, --model-name NAME     Set the model name. Required."
    echo ""
}

validate_directory() {
    local path=$1
    local var_name=$2
    if [ ! -d "$path" ]; then
        echo -e "${RED}Error: Directory for $var_name does not exist: $path${RESET}"
        exit 1
    fi
}

initialize_variables() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -m|--model-name)
                MODEL_NAME="${2,,}" # Convert to lowercase
                shift 2
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${RESET}"
                print_usage
                exit 1
                ;;
        esac
    done

    # Validate that required variables are set
    echo ""
    echo -e "${GREEN}Required flags:${RESET}"
    if [ ! -z "${MODEL_NAME}" ]; then
        echo -e "   ${GREEN}$(printf "%-${COLUMN_WIDTH}s" "MODEL_NAME:") ${MODEL_NAME}${RESET}"
    else
        echo -e "${RED}MODEL_NAME is required but not set${RESET}"
        print_usage
        exit 1
    fi
    
    # Updates to variables which require model name
    MODEL_WEIGHTS_CACHE_PATH="${MODEL_WEIGHTS_CACHE_PATH}/czbenchmarks-${MODEL_NAME}"

    # Docker paths -- should not be changed
    RAW_INPUT_DIR_PATH_DOCKER=/raw
    MODEL_WEIGHTS_PATH_DOCKER=/weights
    MODEL_CODE_PATH_DOCKER=/app
    if [ ! -z "${DEVELOPMENT_CODE_PATH}" ]; then
        BENCHMARK_CODE_PATH_DOCKER=/app/package # Squash container code when mounting local code
    fi

    # # Alternatively, Docker paths can also be loaded from czbenchmarks.constants.py to ensure consistency
    # PYTHON_SCRIPT="from czbenchmarks.constants import RAW_INPUT_DIR_PATH_DOCKER, MODEL_WEIGHTS_PATH_DOCKER; 
    # print(f'RAW_INPUT_DIR_PATH_DOCKER={RAW_INPUT_DIR_PATH_DOCKER}; MODEL_WEIGHTS_PATH_DOCKER={MODEL_WEIGHTS_PATH_DOCKER}')"
    # eval "$(python3 -c "${PYTHON_SCRIPT}")"
}

get_docker_image_uri() {
    # Get model image URI from models.yaml
    MODEL_CONFIG_PATH="conf/models.yaml"
    PYTHON_SCRIPT="import yaml; print(yaml.safe_load(open('${MODEL_CONFIG_PATH}'))['models']['${MODEL_NAME^^}']['model_image_uri'])"
    CZBENCH_CONTAINER_URI=$(python3 -c "${PYTHON_SCRIPT}")

    if [ -z "$CZBENCH_CONTAINER_URI" ]; then
        echo -e "${RED}Model ${MODEL_NAME^^} not found in ${MODEL_CONFIG_PATH}${RESET}"
        exit 1
    fi

    CZBENCH_CONTAINER_NAME=$(basename ${CZBENCH_CONTAINER_URI} | tr ':' '-')
}

print_variables() {
    # Show image information
    echo ""
    echo -e "${GREEN}Docker setup:${RESET}"
    echo -e "   ${GREEN}$(printf "%-${COLUMN_WIDTH}s" "Image:") ${CZBENCH_CONTAINER_URI}${RESET}"
    echo -e "   ${GREEN}$(printf "%-${COLUMN_WIDTH}s" "Container name:") ${CZBENCH_CONTAINER_NAME}${RESET}"

    # Validate required paths and show sources
    echo ""
    echo -e "${GREEN}Local paths:${RESET}"
    for var in DATASETS_CACHE_PATH MODEL_WEIGHTS_CACHE_PATH; do
        if [ -z "${!var}" ]; then
            echo -e "${RED}Error: $var is required but not set${RESET}"
            exit 1
        fi

        validate_directory "${!var}" "$var"
        echo -e "   ${GREEN}$(printf "%-${COLUMN_WIDTH}s" "${var}:") ${!var}${RESET}"
    done

    # Show Docker paths
    echo ""
    echo -e "${GREEN}Docker paths:${RESET}"
    for var in RAW_INPUT_DIR_PATH_DOCKER MODEL_WEIGHTS_PATH_DOCKER MODEL_CODE_PATH_DOCKER; do
        echo -e "   ${GREEN}$(printf "%-${COLUMN_WIDTH}s" "${var}:") ${!var}${RESET}"
    done

    # Development mode
    echo ""
    echo -e "${GREEN}Development mode:${RESET}"
    if [ ! -z "${DEVELOPMENT_CODE_PATH}" ]; then
        validate_directory "${DEVELOPMENT_CODE_PATH}" "DEVELOPMENT_CODE_PATH"
        echo -e "   ${GREEN}$(printf "%-${COLUMN_WIDTH}s" "DEVELOPMENT_CODE_PATH:") ${DEVELOPMENT_CODE_PATH}${RESET}"
        echo -e "   ${GREEN}$(printf "%-${COLUMN_WIDTH}s" "BENCHMARK_CODE_PATH_DOCKER:") ${BENCHMARK_CODE_PATH_DOCKER}${RESET}"
    else
        echo -e "   ${GREEN}DEVELOPMENT_CODE_PATH is not set. Development mode will not be used.${RESET}"
    fi

    # Show user mode information
    echo ""
    echo -e "${GREEN}User mode:${RESET}"
    if [ "${RUN_AS_ROOT}" = "true" ]; then
        echo -e "   ${GREEN}Container will run as root${RESET}"
    else
        echo -e "   ${GREEN}Container will run as current user (${USER})${RESET}"
    fi

    echo ""
    echo -e "${GREEN}AWS credentials:${RESET}"
    if [ -e ${HOME}/.aws/credentials ]; then
        echo -e "   ${GREEN}Using AWS credentials found in ${HOME}/.aws/credentials${RESET}"
    else
        echo -e "${RED}AWS credentials not found in ${HOME}/.aws/credentials${RESET}"
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

    # Add mount points
    DOCKER_CMD="${DOCKER_CMD}
    --volume ${DATASETS_CACHE_PATH}:${RAW_INPUT_DIR_PATH_DOCKER}:rw \\
    --volume ${MODEL_WEIGHTS_CACHE_PATH}:${MODEL_WEIGHTS_PATH_DOCKER}:rw \\"

    # Add code mounts and PYTHONPATH for development mode
    # NOTE: do not change order, cz-benchmarks mounted last to prevent squashing
    if [ ! -z "${DEVELOPMENT_CODE_PATH}" ]; then
        DOCKER_CMD="${DOCKER_CMD}
    --volume ${DEVELOPMENT_CODE_PATH}/docker/${MODEL_NAME}:${MODEL_CODE_PATH_DOCKER}:rw \\
    --volume ${DEVELOPMENT_CODE_PATH}/examples:${MODEL_CODE_PATH_DOCKER}/examples:rw \\
    --volume ${DEVELOPMENT_CODE_PATH}:${BENCHMARK_CODE_PATH_DOCKER}:rw \\
    --env PYTHONPATH=${MODEL_CODE_PATH_DOCKER}:${BENCHMARK_CODE_PATH_DOCKER}/src \\"
    fi

    # Add AWS credentials if they exist
    if [ -e ${HOME}/.aws/credentials ]; then
        DOCKER_CMD="${DOCKER_CMD}
    --volume ${HOME}/.aws:${BENCHMARK_CODE_PATH_DOCKER}/.aws:ro \\"
    fi

    # Add final options
    DOCKER_CMD="${DOCKER_CMD}
    --env HOME=${MODEL_CODE_PATH_DOCKER} \\
    --workdir ${MODEL_CODE_PATH_DOCKER} \\
    --env MODEL_NAME=${MODEL_NAME} \\
    --name ${CZBENCH_CONTAINER_NAME} \\
    --entrypoint ${EVAL_CMD} \\
    ${CZBENCH_CONTAINER_URI}"
}

print_docker_command() {
    echo ""
    echo -e "${GREEN}Executing docker command:${RESET}"
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
RESET="\033[0m"
RED="\033[31m"

# Check if script is run from correct directory
if [ ! "$(ls | grep -c scripts)" -eq 1 ]; then
    echo ""
    echo -e "${RED}Run this script from root directory. Usage: bash scripts/run_docker.sh -m MODEL_NAME${RESET}"
    echo ""
    print_usage
    exit 1
fi

# Setup variables
initialize_variables "$@"
get_docker_image_uri
print_variables

# Ensure docker container is updated
echo ""
echo -e "${GREEN}Pulling latest image for ${MODEL_NAME}${RESET}"
# docker pull ${CZBENCH_CONTAINER_URI}

# FIXME this is a WAR until container images is published
CZBENCH_CONTAINER_URI="czbenchmarks-scvi:latest"

# Create and execute docker command
DOCKER_CMD=""
build_docker_command
print_docker_command
eval ${DOCKER_CMD}
