# Keep the shared runner in this shell so errexit and its cancel traps still work.
tileops_prepare_test_log () {
    if [[ "${CASE_REPEAT:-1}" != 1 ]]; then
        echo "ERROR: TileOPs CI requires CASE_REPEAT=1; diagnose retries separately" >&2
        return 2
    fi
    export CASE_REPEAT=1
    ci_prepare_failure_reports || return 2
    TILEOPS_TEST_LOG="$(ci_failure_report_dir)/tileops-test.log"
    : > "$TILEOPS_TEST_LOG" || return 2
    rm -f "${TILEOPS_TEST_LOG}.done" || return 2
    exec {TILEOPS_CONSOLE_FD}>&1
    python3 "$SCRIPT_DIR/test_log.py" --follow "$TILEOPS_TEST_LOG" \
        >&"$TILEOPS_CONSOLE_FD" 2>&1 &
    TILEOPS_PROGRESS_PID=$!
    trap 'tileops_finish_test_log "$?"' EXIT
    echo "Running TileOPs once per case; live case summaries below; full output: ${TILEOPS_TEST_LOG}.gz"
}

tileops_finish_test_log () {
    local test_exit="$1" log_exit=0
    trap - EXIT
    set +e
    # Drain progress before archiving, including when the test loop exits early.
    touch "${TILEOPS_TEST_LOG}.done" || {
        log_exit=$?
        kill -TERM "$TILEOPS_PROGRESS_PID" 2>/dev/null
    }
    wait "$TILEOPS_PROGRESS_PID" || log_exit=$?
    rm -f "${TILEOPS_TEST_LOG}.done"
    if [[ "$log_exit" -ne 0 ]]; then
        echo "ERROR: live test progress failed (exit $log_exit)" >&"$TILEOPS_CONSOLE_FD"
    fi
    # An exit from inside the redirected runner (including cancel) must report
    # back to the job console before compressing the log it was writing.
    python3 "$SCRIPT_DIR/test_log.py" "$(ci_failure_report_dir)" \
        >&"$TILEOPS_CONSOLE_FD" 2>&1 || log_exit=$?
    if [[ "$test_exit" -eq 0 && "$log_exit" -ne 0 ]]; then
        test_exit=2
    fi
    ci_cleanup_state "$test_exit" >&"$TILEOPS_CONSOLE_FD" 2>&1
}
