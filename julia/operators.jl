import SpecialFunctions: gamma, lgamma, erf, erfc, beta


import Base.FastMath: sqrt_llvm_fast, neg_float_fast,
    add_float_fast, sub_float_fast, mul_float_fast, div_float_fast, rem_float_fast,
    eq_float_fast, ne_float_fast, lt_float_fast, le_float_fast,
    sign_fast, abs_fast, log_fast, log2_fast, log10_fast, sqrt_fast,
    pow_fast

# Implicitly defined:
#binary: mod
#unary: exp, abs, log1p, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, erf, erfc, gamma, relu, round, floor, ceil, round, sign.

# Use some fast operators from https://github.com/JuliaLang/julia/blob/81597635c4ad1e8c2e1c5753fda4ec0e7397543f/base/fastmath.jl
# Define allowed operators. Any julia operator can also be used.
plus(x::Float64, y::Float64)::Float64 = add_float_fast(x, y) #Do not change the name of this operator.
sub(x::Float64, y::Float64)::Float64 = sub_float_fast(x, y) #Do not change the name of this operator.
mult(x::Float64, y::Float64)::Float64 = mul_float_fast(x, y) #Do not change the name of this operator.
square(x::Float64)::Float64 = mul_float_fast(x, x)
cube(x::Float64)::Float64 = mul_float_fast(mul_float_fast(x, x), x)
pow(x::Float64, y::Float64)::Float64 = sign_fast(x)*pow_fast(abs(x), y)
div(x::Float64, y::Float64)::Float64 = div_float_fast(x, y)
logm(x::Float64)::Float64 = log_fast(abs_fast(x) + 1e-8)
logm2(x::Float64)::Float64 = log2_fast(abs_fast(x) + 1e-8)
logm10(x::Float64)::Float64 = log10_fast(abs_fast(x) + 1e-8)
sqrtm(x::Float64)::Float64 = sqrt_fast(abs_fast(x))
neg(x::Float64)::Float64 = neg_float_fast(x)

function greater(x::Float64, y::Float64)::Float64
    if x > y
        return 1e0
    end
    return 0e0
end

function relu(x::Float64)::Float64
    if x > 0e0
        return x
    end
    return 0e0
end

function logical_or(x::Float64, y::Float64)::Float64
    if x > 0e0 || y > 0e0
        return 1e0
    end
    return 0e0
end

# (Just use multiplication normally)
function logical_and(x::Float64, y::Float64)::Float64
    if x > 0e0 && y > 0e0
        return 1e0
    end
    return 0e0
end
