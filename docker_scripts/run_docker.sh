#!/bin/bash
#
# Interactive docker container launch script for cz-benchmarks
# FIXME: Add README "Run Docker Container in Interactive Mode" section for detailed usage instructions?

# Local mount paths
LOCAL_RAW_INPUT_DIR_PATH="${LOCAL_RAW_INPUT_DIR_PATH:-${LOCAL_RAW_INPUT_DIR_PATH:-${HOME}/.cz-benchmarks/datasets}}" # DATASETS_CACHE_PATH
LOCAL_MODEL_WEIGHTS_PATH="${LOCAL_MODEL_WEIGHTS_PATH:-${LOCAL_MODEL_WEIGHTS_PATH:-${HOME}/.cz-benchmarks/weights}}" # MODEL_WEIGHTS_CACHE_PATH
LOCAL_INPUT_PATH="${local_input_path:-${LOCAL_INPUT_PATH:-${HOME}/.cz-benchmarks/datasets}}"
LOCAL_OUTPUT_PATH="${local_output_path:-${LOCAL_OUTPUT_PATH:-${HOME}/.cz-benchmarks/output}}"
LOCAL_CODE_PATH="${local_code_path:-${LOCAL_CODE_PATH:-$(pwd)}}"

# Container settings
CZBENCH_CONTAINER_NAME="${CZBENCH_CONTAINER_NAME:=czbenchmarks-${MODEL_NAME}}"
CZBENCH_IMG_TAG="${CZBENCH_IMG_TAG:-${CZBENCH_IMG_TAG:latest}}"
EVAL_CMD="${EVAL_CMD:=bash}"
RUN_AS_ROOT="${RUN_AS_ROOT:=false}" # Default to running as current user (not root)
ADDITIONAL_DOCKER_FLAGS="${ADDITIONAL_DOCKER_FLAGS:=}" # Defaults to no additional flags

################################################################################
# Function definitions
# Function to print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -m, --model-name NAME     Set the model name (env: MODEL_NAME). Required."
    echo "  -i, --image NAME          Set the Docker image name (env: CZBENCH_IMG)"
    echo "  -t, --tag TAG             Set the Docker image tag (env: CZBENCH_IMG_TAG)"
    echo "  -d, --dataset-path PATH   Set the dataset path (env: LOCAL_RAW_INPUT_DIR_PATH)"
    echo "      --model-path PATH     Set the model path (env: LOCAL_MODEL_WEIGHTS_PATH)"
    echo "  -r, --results-path PATH   Set the results path (env: LOCAL_OUTPUT_DATA_DIR_DOCKER)"
    echo "  -c, --code-path PATH      Set the code path (env: LOCAL_CODE_PATH)"
    echo "  -e, --eval-cmd CMD        Set the evaluation command (env: EVAL_CMD), will be added to \"bash -c \${CMD}\""
    echo "      --docker-flags FLAGS  Additional Docker flags (env: ADDITIONAL_DOCKER_FLAGS)"
    echo "                            Add an extra set of quotes around the flag value if it contains spaces."
    echo "      --run-as-root         Run container as root instead of current user (env: RUN_AS_ROOT)"

    echo ""
    echo "Each option can be set either via command line flag or environment variable."
    echo "Command line flags take precedence over environment variables."
    echo "See README.md for more detailed instructions."
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
    # Initialize variables as empty
    local czbench_img=""
    local czbench_img_tag=""
    local local_raw_input_dir_path=""
    local local_model_weights_path=""
    local local_output_path=""
    local local_input_path=""
    local local_code_path=""
    local eval_cmd=""
    local additional_docker_flags=""
    local run_as_root=""
    local czbench_img_source="environment variable"
    local czbench_img_tag_source="environment variable"
    local local_raw_input_dir_source="environment variable"
    local local_model_weights_source="environment variable"
    local local_output_source="environment variable"
    local local_input_source="environment variable"
    local local_code_source="environment variable"
    local additional_docker_flags_source="environment variable"
    local run_as_root_source="environment variable"

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -m|--model-name)
                MODEL_NAME="${2,,}" # Convert to lowercase
                shift 2
                ;;
               --image)
                czbench_img="$2"
                czbench_img_source="command line flag"
                shift 2
                ;;
               --tag)
                czbench_img_tag="$2"
                czbench_img_tag_source="command line flag"
                shift 2
                ;;
            -d|--dataset-path)
                local_raw_input_dir_path="$2"
                local_raw_input_dir_source="command line flag"
                shift 2
                ;;
            --model-path)
                local_model_weights_path="$2"
                local_model_weights_source="command line flag"
                shift 2
                ;;
            -i|--input-path)
                local_input_path="$2"
                local_input_source="command line flag"
                shift 2
                ;;
            -o|--output-path)
                local_output_path="$2"
                local_output_source="command line flag"
                shift 2
                ;;
            -c|--code-path)
                local_code_path="$2"
                local_code_source="command line flag"
                shift 2
                ;;
            -e|--eval-cmd)
                eval_cmd="bash -c \"$2\""
                shift 2
                ;;
            --docker-flags)
                additional_docker_flags="$2"
                additional_docker_flags_source="command line flag"
                shift 2
                ;;
            --run-as-root)
                run_as_root="true"
                run_as_root_source="command line flag"
                shift
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
    if [ ! -z "${MODEL_NAME}" ]; then
        echo -e "${GREEN}MODEL_NAME is set to ${MODEL_NAME}${RESET}"
    else
        echo -e "${RED}MODEL_NAME is required but not set${RESET}"
        print_usage
        exit 1
    fi

    # Variables that can be set from command line or environment
    CZBENCH_IMG="${czbench_img:-${CZBENCH_IMG}}"
    CZBENCH_IMG_TAG="${czbench_img_tag:-${CZBENCH_IMG_TAG}}"
    EVAL_CMD="${eval_cmd:-${EVAL_CMD}}"
    ADDITIONAL_DOCKER_FLAGS="${additional_docker_flags:-${ADDITIONAL_DOCKER_FLAGS}}"
    RUN_AS_ROOT="${run_as_root:-${RUN_AS_ROOT}}"
    LOCAL_RAW_INPUT_DIR_PATH="${local_raw_input_dir_path:-${LOCAL_RAW_INPUT_DIR_PATH}}"
    LOCAL_MODEL_WEIGHTS_PATH="${local_model_weights_path:-${LOCAL_MODEL_WEIGHTS_PATH}}"
    LOCAL_INPUT_PATH="${local_input_path:-${LOCAL_INPUT_PATH}}"
    LOCAL_OUTPUT_PATH="${local_output_path:-${LOCAL_OUTPUT_PATH}}"
    LOCAL_CODE_PATH="${local_code_path:-${LOCAL_CODE_PATH}}"

    # Updates to paths
    LOCAL_MODEL_WEIGHTS_PATH="${LOCAL_MODEL_WEIGHTS_PATH}/czbenchmarks-${MODEL_NAME}"

    # Show image information
    echo ""
    echo -e "${GREEN}Docker image set to ${CZBENCH_IMG}:${CZBENCH_IMG_TAG} (image from ${czbench_img_source}, tag from ${czbench_img_tag_source})${RESET}"

    # Validate required paths and show sources
    for var in LOCAL_RAW_INPUT_DIR_PATH LOCAL_MODEL_WEIGHTS_PATH LOCAL_INPUT_PATH LOCAL_OUTPUT_PATH; do
        local source_var="${var,,}"              # First convert to lowercase
        source_var="${source_var/_path/_source}" # Then replace _path with _source

        if [ -z "${!var}" ]; then
            echo -e "${RED}Error: $var is not set (via environment variable or command line flag)${RESET}"
            exit 1
        fi

        validate_directory "${!var}" "$var"
        echo -e "${GREEN}${var} is set to ${!var} (from ${!source_var})${RESET}"
    done

    # Handle code path for development mode
    echo ""
    if [ ! -z "${LOCAL_CODE_PATH}" ]; then
        validate_directory "${LOCAL_CODE_PATH}" "LOCAL_CODE_PATH"
        echo -e "${GREEN}Using development mode${RESET}"
        echo -e "${GREEN}LOCAL_CODE_PATH is set to ${LOCAL_CODE_PATH} (from $local_code_source)${RESET}"
    else
        echo -e "${GREEN}LOCAL_CODE_PATH is not set. Development mode will not be used.${RESET}"
    fi
    
    # Show additional Docker flags if set
    if [ ! -z "${ADDITIONAL_DOCKER_FLAGS}" ]; then
        echo -e "${GREEN}Additional Docker flags: ${ADDITIONAL_DOCKER_FLAGS} (from ${additional_docker_flags_source})${RESET}"
    fi

    # Show user mode information
    if [ "${RUN_AS_ROOT}" = "true" ]; then
        echo -e "${GREEN}Container will run as root (from ${run_as_root_source})${RESET}"
    else
        echo -e "${GREEN}Container will run as current user (${USER}) (from ${run_as_root_source})${RESET}"
    fi
}

################################################################################
# Main script execution starts here

# ANSI color codes
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

# Docker Paths -- should not be changed 
CODE_PATH=/app/package
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

for var in RAW_INPUT_DIR_PATH_DOCKER MODEL_WEIGHTS_PATH_DOCKER INPUT_DATA_DIR_DOCKER OUTPUT_DATA_DIR_DOCKER; do
    echo -e "${GREEN}${var} is set to ${!var} (from czbenchmarks.constants.py)${RESET}"
done

# Call setup_variables with all command line arguments
setup_variables "$@"

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

# Add user-specific settings if not running as root
if [ "${RUN_AS_ROOT}" != "true" ]; then
    DOCKER_CMD="${DOCKER_CMD}
--volume /etc/passwd:/etc/passwd:ro \\
--volume /etc/group:/etc/group:ro \\
--volume /etc/shadow:/etc/shadow:ro \\
--user $(id -u):$(id -g) \\
--volume ${HOME}/.ssh:${HOME}/.ssh:ro \\"
fi

# Add dataset mount and environment variables
DOCKER_CMD="${DOCKER_CMD}
--volume ${LOCAL_RAW_INPUT_DIR_PATH}:${RAW_INPUT_DIR_PATH_DOCKER}:rw \\
--volume ${LOCAL_MODEL_WEIGHTS_PATH}:${MODEL_WEIGHTS_PATH_DOCKER}:rw \\
--volume ${LOCAL_INPUT_PATH}:${INPUT_DATA_DIR_DOCKER}:rw \\
--volume ${LOCAL_OUTPUT_PATH}:${OUTPUT_DATA_DIR_DOCKER}:rw \\"

# Add code mount and PYTHONPATH for development mode
# FIXME: is there a better solution to ensure code can be imported from both src and docker/MODEL_NAME? 
if [ ! -z "${LOCAL_CODE_PATH}" ]; then
    DOCKER_CMD="${DOCKER_CMD}
--volume ${LOCAL_CODE_PATH}:${CODE_PATH}:rw \\
--env PYTHONPATH=${CODE_PATH}/src:${CODE_PATH}/docker/${MODEL_NAME}:${CODE_PATH}:"'$PYTHONPATH'" \\"
fi

# Add additional Docker flags if provided
if [ ! -z "${ADDITIONAL_DOCKER_FLAGS}" ]; then
    DOCKER_CMD="${DOCKER_CMD}
${ADDITIONAL_DOCKER_FLAGS} \\"
fi

# Add AWS credentials if they exist
if [ -e ${HOME}/.aws/credentials ]; then
    DOCKER_CMD="${DOCKER_CMD}
--volume ${HOME}/.aws/credentials:${CODE_PATH}/.aws/credentials:ro \\"
else
    echo -e "${RED}AWS credentials not added because they were not found in ${HOME}/.aws/credentials${RESET}"
fi

# Add final options
DOCKER_CMD="${DOCKER_CMD}
--env HOME=${CODE_PATH} \\
--workdir ${CODE_PATH} \\
--env MODEL_NAME=${MODEL_NAME} \\
--name ${CZBENCH_CONTAINER_NAME} \\
--entrypoint ${EVAL_CMD} \\
${CZBENCH_IMG}:${CZBENCH_IMG_TAG}"

# Print the full command
echo ""
echo -e "${GREEN}Executing docker command:${RESET}"
echo "${DOCKER_CMD}"
echo ""

# Remove line continuations before execution
DOCKER_CMD_EVAL=$(echo "${DOCKER_CMD}" | tr -d '\\')
eval ${DOCKER_CMD_EVAL}
