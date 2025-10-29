
# special_arg="$1"        # first argument to handle separately
# shift                   # remove the first argument from $@

# # now $@ contains only the remaining arguments
# echo "Handling special argument: $special_arg"
# echo "Passing the rest: $@"


julia-x86_64 --color=yes --project $@ make.jl && \
rm -rf /project/rokpintar/public_html/polfed && \
cp -R build /project/rokpintar/public_html/polfed
