#!/bin/bash
#
# Interactive docker container launch script for cz-benchmarks
# FIXME: Add README "Run Docker Container in Interactive Mode" section for detailed usage instructions?

# Local mount paths
# FIXME: should input / output paths be mounted since they could contain stale files? Ensure they are empty?
DATASETS_CACHE_PATH=${HOME}/.cz-benchmarks/datasets
MODEL_WEIGHTS_CACHE_PATH=${HOME}/.cz-benchmarks/weights
INPUT_CACHE_PATH=${HOME}/.cz-benchmarks/datasets
OUTPUT_CACHE_PATH=${HOME}/.cz-benchmarks/output
LOCAL_CODE_PATH=$(pwd) # Leave blank or remove to not mount code

# Container settings
EVAL_CMD=bash # e.g. bash or '''python3 example_interactive.py'''
RUN_AS_ROOT=false # false or true

################################################################################
# Function definitions
# Function to print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -m, --model-name NAME     Set the model name. Required."
    echo ""
}

# Function to validate that directory exists
validate_directory() {
    local path=$1
    local var_name=$2
    if [ ! -d "$path" ]; then
        echo -e "${RED}Error: Directory for $var_name does not exist: $path${RESET}"
        exit 1
    fi
}

# Function to process variables
setup_variables() {
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

    # Required variables
    echo ""
    echo -e "${GREEN}Required flags:${RESET}"
    if [ ! -z "${MODEL_NAME}" ]; then
        echo -e "   ${GREEN}$(printf "%-${COLUMN_WIDTH:-15}s" "MODEL_NAME:") ${MODEL_NAME}${RESET}"
    else
        echo -e "${RED}MODEL_NAME is required but not set${RESET}"
        print_usage
        exit 1
    fi
    
    # Updates to variables which require model name
    CZBENCH_IMG="czbenchmarks-${MODEL_NAME}"
    CZBENCH_IMG_TAG="latest"
    MODEL_WEIGHTS_CACHE_PATH="${MODEL_WEIGHTS_CACHE_PATH}/czbenchmarks-${MODEL_NAME}"
}

print_variables() {
    # Show image information
    echo ""
    echo -e "${GREEN}Docker setup:${RESET}"
    echo -e "   ${GREEN}$(printf "%-${COLUMN_WIDTH:-15}s" "Docker image:") ${CZBENCH_IMG}:${CZBENCH_IMG_TAG}${RESET}"

    # Validate required paths and show sources
    echo ""
    echo -e "${GREEN}Local paths:${RESET}"
    for var in DATASETS_CACHE_PATH MODEL_WEIGHTS_CACHE_PATH INPUT_CACHE_PATH OUTPUT_CACHE_PATH; do
        if [ -z "${!var}" ]; then
            echo -e "${RED}Error: $var is required but not set${RESET}"
            exit 1
        fi

        validate_directory "${!var}" "$var"
        echo -e "   ${GREEN}$(printf "%-${COLUMN_WIDTH:-15}s" "${var}:") ${!var}${RESET}"
    done

    # Handle code path for development mode
    echo ""
    echo -e "${GREEN}Development mode:${RESET}"
    if [ ! -z "${LOCAL_CODE_PATH}" ]; then
        validate_directory "${LOCAL_CODE_PATH}" "LOCAL_CODE_PATH"
        echo -e "   ${GREEN}Using development mode."
        echo -e "   ${GREEN}$(printf "%-${COLUMN_WIDTH:-15}s" "LOCAL_CODE_PATH:") ${LOCAL_CODE_PATH}${RESET}"
    else
        echo -e "   ${GREEN}LOCAL_CODE_PATH is not set. Development mode will not be used.${RESET}"
    fi

    # Show Docker paths
    echo ""
    echo -e "${GREEN}Docker paths:${RESET}"
    for var in RAW_INPUT_DIR_PATH_DOCKER MODEL_WEIGHTS_PATH_DOCKER INPUT_DATA_DIR_DOCKER OUTPUT_DATA_DIR_DOCKER; do
        echo -e "   ${GREEN}$(printf "%-${COLUMN_WIDTH:-15}s" "${var}:") ${!var}${RESET}"
    done

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
    --volume ${MODEL_WEIGHTS_CACHE_PATH}:${MODEL_WEIGHTS_PATH_DOCKER}:rw \\
    --volume ${INPUT_CACHE_PATH}:${INPUT_DATA_DIR_DOCKER}:rw \\
    --volume ${OUTPUT_CACHE_PATH}:${OUTPUT_DATA_DIR_DOCKER}:rw \\"

    # Add code mount and PYTHONPATH for development mode
    # FIXME: better solution to ensure code can be imported from both src and docker/MODEL_NAME? 
    if [ ! -z "${LOCAL_CODE_PATH}" ]; then
        DOCKER_CMD="${DOCKER_CMD}
    --volume ${LOCAL_CODE_PATH}:${CODE_PATH_DOCKER}:rw \\
    --env PYTHONPATH=${CODE_PATH_DOCKER}/src:${CODE_PATH_DOCKER}/docker/${MODEL_NAME}:${CODE_PATH_DOCKER}:"'$PYTHONPATH'" \\"
    fi

    # Add AWS credentials if they exist
    if [ -e ${HOME}/.aws/credentials ]; then
        DOCKER_CMD="${DOCKER_CMD}
    --volume ${HOME}/.aws/credentials:${CODE_PATH_DOCKER}/.aws/credentials:ro \\"
    fi

    # Add final options
    DOCKER_CMD="${DOCKER_CMD}
    --env HOME=${CODE_PATH_DOCKER} \\
    --workdir ${CODE_PATH_DOCKER} \\
    --env MODEL_NAME=${MODEL_NAME} \\
    --name ${CZBENCH_CONTAINER_NAME} \\
    --entrypoint ${EVAL_CMD} \\
    ${CZBENCH_IMG}:${CZBENCH_IMG_TAG}"
}

################################################################################
# Main script execution starts here

# Docker Paths -- should not be changed 
CODE_PATH_DOCKER=/app/package
RAW_INPUT_DIR_PATH_DOCKER=/raw
MODEL_WEIGHTS_PATH_DOCKER=/weights
INPUT_DATA_PATH_DOCKER=/input/data.dill
OUTPUT_DATA_PATH_DOCKER=/output/data.dill

# Can also be loaded from constants.py
# PYTHON_SCRIPT="from czbenchmarks.constants import RAW_INPUT_DIR_PATH_DOCKER, MODEL_WEIGHTS_PATH_DOCKER, INPUT_DATA_PATH_DOCKER, OUTPUT_DATA_PATH_DOCKER; 
# print(f'RAW_INPUT_DIR_PATH_DOCKER={RAW_INPUT_DIR_PATH_DOCKER}; MODEL_WEIGHTS_PATH_DOCKER={MODEL_WEIGHTS_PATH_DOCKER}; INPUT_DATA_PATH_DOCKER={INPUT_DATA_PATH_DOCKER}; OUTPUT_DATA_PATH_DOCKER={OUTPUT_DATA_PATH_DOCKER}')"
# eval "$(python3 -c "${PYTHON_SCRIPT}")" # Get Docker paths from constants.py

INPUT_DATA_DIR_DOCKER=$(dirname ${INPUT_DATA_PATH_DOCKER})
OUTPUT_DATA_DIR_DOCKER=$(dirname ${OUTPUT_DATA_PATH_DOCKER})

# Print formatting
COLUMN_WIDTH=30
GREEN="\033[32m"
RESET="\033[0m"
RED="\033[31m"

# Check if script is run from correct directory
if [ ! "$(ls | grep -c docker_scripts)" -eq 1 ]; then
    echo ""
    echo -e "${RED}Run this script from root directory. Usage: bash ./docker_scripts/run_docker_interactive.sh${RESET}"
    echo ""
    print_usage
    exit 1
fi

# Setup variables
setup_variables "$@"
print_variables


# Create docker command
build_docker_command

echo ""
echo -e "${GREEN}Executing docker command:${RESET}"
echo "${DOCKER_CMD}"
echo ""

# Remove extra line continuations for printing before execution
DOCKER_CMD=$(echo "${DOCKER_CMD}" | tr -d '\\')
eval ${DOCKER_CMD}
