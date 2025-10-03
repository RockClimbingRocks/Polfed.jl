"""
    Formatter

A simple type for controlling conditional string formatting, such as colored or bold text output.
The `color` field determines whether formatting codes are applied.

# Fields
- `color::Bool`: If `true`, enables formatting codes; otherwise, outputs plain text.
"""
struct Formatter
    color::Bool
end



"""
    fmt(f::Formatter, str::AbstractString; code="")

Conditionally formats the string `str` using ANSI escape codes if `f.color` is `true` and `code` is not empty.
If formatting is disabled, returns the original string.

# Arguments
- `f::Formatter`: The formatter instance.
- `str::AbstractString`: The string to format.
- `code`: (Optional) The ANSI code to use for formatting.

# Returns
- `String`: The formatted or plain string.
"""
function fmt(f::Formatter, str::AbstractString; code="")
    f.color && !isempty(code) ? "\e[$(code)m$(str)\e[0m" : str
end


bold(f::Formatter, s)  = fmt(f, s; code="1")
cyan(f::Formatter, s)  = fmt(f, s; code="1;36")
blue(f::Formatter, s)  = fmt(f, s; code="1;34")
green(f::Formatter, s) = fmt(f, s; code="1;32")
red(f::Formatter, s)   = fmt(f, s; code="1;31")
yellow(f::Formatter, s)= fmt(f, s; code="1;33")
