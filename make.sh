make_arg=""
debug_args="d debug"

for arg in $@; do
    for variant in $debug_args; do
        if [ "$arg" = "$variant" ] || [ "$arg" = "-$variant" ]; then
            make_arg="debug"
        fi
    done
done

make $make_arg
