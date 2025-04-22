log_command() {
    local start_time end_time elapsed

    start_time=$(date +%s)

    {
	echo "BENCHMARK_RUN_COMMAND:"
	echo
        echo "\$ $*"
        echo
        echo
	echo "BENCHMARK_RUN_LOGS:"
	echo
        "$@"
        exit_code=$?
        echo
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "⏱️ Command completed in ${elapsed}s (exit code: $exit_code)"
        echo
        echo
    } | tee -a benchmark_run_output.txt
}

extract_model_logs() {
    local input_file="$1"
    local output_file="${2:-benchmark_results.txt}"

    if [[ ! -f "$input_file" ]]; then
        echo "Error: '$input_file' does not exist."
        return 1
    fi

    awk '
    /Model ran successfully/ { capture=1 }
    /BENCHMARK_RUN_COMMAND:/ && capture { capture=0; next }
    capture
    ' "$input_file" > "$output_file"

    echo "Extracted logs saved to '$output_file'"
}
