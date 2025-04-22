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
extract_benchmark_runs() {
    local input_file="$1"
    local output_file="${2:-benchmark_results.txt}"

    if [[ ! -f "$input_file" ]]; then
        echo "Error: File '$input_file' does not exist."
        return 1
    fi

    awk '
    /BENCHMARK_RUN_COMMAND:/ {
        in_section = 0   # Reset any active capture on new benchmark start
    }

    /args: czbenchmarks run/ {
        capture = 1      # Start capturing from here
    }

    /Command completed/ && capture {
        print $0         # Include this line
        print ""         # Add spacing between iterations
        capture = 0      # End capture
        next
    }

    capture {
        print $0
    }
    ' "$input_file" > "$output_file"

    echo "Extracted benchmark logs saved to: $output_file"
}
