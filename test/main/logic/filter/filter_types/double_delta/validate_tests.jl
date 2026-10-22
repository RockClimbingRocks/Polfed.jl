# Leakage validation for the DoubleDeltaFilter (validate.jl). Ground truth is the combined polynomial
# P: within each passband P is past its cutoff (amplified), and in the three suppression regions
# (left tail / valley / right tail) P stays inside the band [cutoff_lower, cutoff_upper]. A "leak" is
# any escape from that contract.
#
# The corpus below is bucketed by WHERE the leak is, written as  left-tail | valley | right-tail:
#   good_filters      GOOD | GOOD | GOOD   (no leak anywhere)
#   bad_left          BAD  | GOOD | GOOD
#   bad_valley        GOOD | BAD  | GOOD
#   bad_right         GOOD | GOOD | BAD
#   bad_left_valley   BAD  | BAD  | GOOD
#   bad_left_right    BAD  | GOOD | BAD
#   bad_valley_right  GOOD | BAD  | BAD
#   bad_all           BAD  | BAD  | BAD
# This lets each region check (validate_outer_tails / validate_between_passbands) be exercised on
# exactly the buckets it should pass or fail. In EVERY filter the passband interiors are clean, so
# validate_passbands must be true for all of them.
#
# NOTE: filters are passed directly (all-fields constructor, no build_filter). If a case fails — a
# region marked BAD that validate reports clean, or vice versa — it is LEFT in place (not silently
# dropped) so the categorization vs. validator can be re-examined.

using Test
using Polfed
using Polfed: Interval   # so filters can be written directly, without build_filter

# left | valley | right  =  GOOD | GOOD | GOOD
good_filters = [
    Polfed.DoubleDeltaFilter(Interval(-0.999, -0.998), Interval(-0.9990279262803854, -0.9978010596342526), -0.9984999999999999, 227, Interval(-0.9, -0.85), Interval(-0.9017057255195082, -0.8444590076236276), -0.875, 42, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.8, -0.75), Interval(-0.8015362678680912, -0.7470946725888851), -0.775, 58, Interval(0.2, 0.22), Interval(0.1994844364130072, 0.22036298588184264), 0.21000000000000002, 230, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.2, -0.1), Interval(-0.20279851395303147, -0.09398520953992376), -0.15000000000000002, 38, Interval(0.1, 0.2), Interval(0.09398520953992388, 0.20279851395303158), 0.15000000000000002, 38, -0.35, 0.35, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.02, -0.01), Interval(-0.020155049637246444, -0.009829248362306342), -0.015, 522, Interval(0.05, 0.2), Interval(0.04286511446536206, 0.20288149628363433), 0.125, 32, -0.2, 0.35, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.12, -0.01), Interval(-0.12229514949186684, 0.010190313131929547), -0.065, 37, Interval(0.05, 0.2), Interval(0.038272886028551986, 0.20252928217256247), 0.125, 35, -0.2, 0.35, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.02, -0.01), Interval(-0.020316670412413657, -0.00983008910624936), -0.015, 474, Interval(0.05, 0.06), Interval(0.049825669542163956, 0.060205640347757494), 0.055, 490, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.12, -0.11), Interval(-0.12034345300763047, -0.10986424549667022), -0.11499999999999999, 462, Interval(-0.06, -0.05), Interval(-0.060154316536131174, -0.04963038732380819), -0.055, 474, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(0.238, 0.2489), Interval(0.2378364076079442, 0.24905981258608503), 0.24345, 427, Interval(0.42, 0.419), Interval(0.4189845441232803, 0.42001552620279436), 0.4195, 4397, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.5, -0.4), Interval(-0.503262639415271, -0.39424281398156097), -0.45, 40, Interval(0.95, 0.975), Interval(0.9468546962014138, 0.9753230966475539), 0.9624999999999999, 46, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.5, -0.4), Interval(-0.502678827479967, -0.39921417578887697), -0.45, 41, Interval(0.9, 0.975), Interval(0.8761082045655879, 0.9756311709131562), 0.9375, 17, -0.25, 0.3, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.7, -0.4), Interval(-0.7061950787870496, -0.3848573813579976), -0.55, 11, Interval(0.9, 0.975), Interval(0.8662054413703038, 0.9798959263178314), 0.9375, 16, -0.25, 0.3, 0.97),
]

# BAD | GOOD | GOOD  (left tail leaks)
bad_left = [
    Polfed.DoubleDeltaFilter(Interval(-0.2, -0.1), Interval(-0.2036562416091198, -0.0974226650145714), -0.15000000000000002, 45, Interval(0.1, 0.2), Interval(0.09409181936582067, 0.20165938573567713), 0.15000000000000002, 48, -0.25, 0.35, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.41, -0.19), Interval(-0.4126959952037126, -0.14301345834952106), -0.3, 18, Interval(0.09, 0.51), Interval(0.028809243333375634, 0.5471779108569886), 0.3, 9, -0.25, 0.25, 0.97),
]

# GOOD | BAD | GOOD  (valley leaks)
bad_valley = [
    Polfed.DoubleDeltaFilter(Interval(-0.999, -0.998), Interval(-0.9990298141606104, -0.9978245460037383), -0.9984999999999999, 223, Interval(-0.97, -0.95), Interval(-0.9705795295129905, -0.9480653216279992), -0.96, 64, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.2, -0.1), Interval(-0.2026194152170053, -0.09056031393637208), -0.15000000000000002, 42, Interval(0.1, 0.2), Interval(0.09056031393637176, 0.2026194152170054), 0.15000000000000002, 42, -0.3, 0.3, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.12, -0.11), Interval(-0.12035249778377452, -0.10987597407253742), -0.11499999999999999, 448, Interval(-0.08, -0.05), Interval(-0.0802990799427931, -0.04906627251952418), -0.065, 159, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.012, -0.011), Interval(-0.012035083701174577, -0.010984201036823175), -0.0115, 4498, Interval(-0.008, -0.005), Interval(-0.008031163070447014, -0.0048990937837169015), -0.006500000000000001, 1594, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(0.238, 0.339), Interval(0.2373493497052531, 0.3439796679458178), 0.2885, 45, Interval(0.67, 0.749), Interval(0.6658785490571112, 0.7517895437780161), 0.7095, 39, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.5, -0.4), Interval(-0.5008587072317201, -0.396826570116449), -0.45, 43, Interval(0.95, 0.975), Interval(0.9464193913942973, 0.9753568545574903), 0.9624999999999999, 46, -0.245, 0.225, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.5, -0.4), Interval(-0.503262639415271, -0.39424281398156097), -0.45, 40, Interval(0.95, 0.975), Interval(0.9467753847616527, 0.9753813363671952), 0.9624999999999999, 46, -0.245, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.5, -0.4), Interval(-0.5016181893265156, -0.3995478309598389), -0.45, 43, Interval(0.9, 0.975), Interval(0.8752933238702808, 0.9758223458809451), 0.9375, 17, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.5, -0.4), Interval(-0.5013333137086337, -0.3956915521907782), -0.45, 40, Interval(0.22499999999999998, 0.375), Interval(0.22297621648694613, 0.3762270091364251), 0.3, 31, -0.25, 0.25, 0.97),
]

# GOOD | GOOD | BAD  (right tail leaks)
bad_right = [
    Polfed.DoubleDeltaFilter(Interval(-0.2, -0.1), Interval(-0.20183461129515706, -0.0978655971928905), -0.15000000000000002, 49, Interval(0.1, 0.2), Interval(0.0975326294146677, 0.20418241608110607), 0.15000000000000002, 47, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.12, -0.01), Interval(-0.12265759766300893, 0.009550227423602127), -0.065, 43, Interval(0.05, 0.2), Interval(0.03478064774764691, 0.2031271369014796), 0.125, 35, -0.2, 0.2, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.4, -0.19999999999999998), Interval(-0.4023261311339191, -0.19177847413648613), -0.3, 24, Interval(0.19, 0.21000000000000002), Interval(0.1891845315197194, 0.2102943063542795), 0.2, 235, -0.25, 0.175, 0.97),
]

# BAD | BAD | GOOD  (left tail + valley leak)
bad_left_valley = [
    Polfed.DoubleDeltaFilter(Interval(-0.2, -0.1), Interval(-0.20441080777080978, -0.0971106071287536), -0.15000000000000002, 43, Interval(0.1, 0.2), Interval(0.09404850253469642, 0.2001612249525239), 0.15000000000000002, 49, -0.2, 0.35, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.02, -0.01), Interval(-0.020324694793762064, -0.009824100931059442), -0.015, 401, Interval(0.1, 0.2), Interval(0.09967516256540035, 0.2016795970120905), 0.15000000000000002, 49, -0.2, 0.35, 0.97),
    Polfed.DoubleDeltaFilter(Interval(0.968, 0.969), Interval(0.9679860529941575, 0.969027774091956), 0.9684999999999999, 1125, Interval(0.972, 0.975), Interval(0.971906160802671, 0.9750389186658114), 0.9735, 357, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(0.983, 0.984), Interval(0.9829806587885976, 0.98410023117576), 0.9835, 790, Interval(0.987, 0.99), Interval(0.9868175700564817, 0.9900657538789703), 0.9884999999999999, 224, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.5, -0.4), Interval(-0.5013333137086337, -0.3956915521907782), -0.45, 40, Interval(0.22499999999999998, 0.375), Interval(0.22196156768868647, 0.377400420391631), 0.3, 31, -0.235, 0.25, 0.97),
]

# BAD | GOOD | BAD  (both tails leak)
bad_left_right = [
    Polfed.DoubleDeltaFilter(Interval(-0.3, -0.2), Interval(-0.30473074862800686, -0.19766335524224388), -0.25, 44, Interval(0.2, 0.3), Interval(0.19766335524224402, 0.304730748628007), 0.25, 44, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.035, -0.034), Interval(-0.035094135525995405, -0.03396150892460559), -0.0345, 4616, Interval(-0.033, -0.032), Interval(-0.03301952151546298, -0.03191103795637813), -0.0325, 4445, -0.245, 0.225, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.9, -0.85), Interval(-0.9003595605571001, -0.8476232407144195), -0.875, 46, Interval(0.85, 0.9), Interval(0.8476232407144192, 0.9003595605571003), 0.875, 46, -0.22, 0.22, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.5, -0.4), Interval(-0.5109146267495679, -0.37961352623080297), -0.45, 41, Interval(-0.33, -0.2), Interval(-0.3375338378978964, -0.19795651238362288), -0.265, 39, -0.245, 0.225, 0.97),
]

# GOOD | BAD | BAD  (valley + right tail leak)
bad_valley_right = [
    Polfed.DoubleDeltaFilter(Interval(0.28800000000000003, 0.489), Interval(0.2869030771148399, 0.4805052615230744), 0.3885, 23, Interval(0.492, 0.499), Interval(0.4918808332287643, 0.49936674724381525), 0.4955, 524, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(0.28800000000000003, 0.489), Interval(0.27832213912850495, 0.4828642753473208), 0.3885, 21, Interval(0.492, 0.495), Interval(0.49195357957501906, 0.4950872383493594), 0.4935, 1202, -0.25, 0.25, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.35, -0.34), Interval(-0.35013334053642176, -0.3398338806753482), -0.345, 457, Interval(-0.33, -0.331), Interval(-0.3310357805516722, -0.32998578577374177), -0.3305, 4178, -0.25, 0.25, 0.97),
    # middle leak is very hard to see: the difference is ~1e-5
    Polfed.DoubleDeltaFilter(Interval(-0.35, -0.34), Interval(-0.35018784355928945, -0.339852057219952), -0.345, 445, Interval(-0.33, -0.331), Interval(-0.33104269786798063, -0.3299848148026443), -0.3305, 4121, -0.27, 0.27, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.5, -0.4), Interval(-0.501456287938313, -0.3946890444896281), -0.45, 43, Interval(-0.33, -0.3), Interval(-0.33408323005299617, -0.2995609631958611), -0.315, 138, -0.245, 0.225, 0.97),
]

# BAD | BAD | BAD  (all three regions leak)
bad_all = [
    Polfed.DoubleDeltaFilter(Interval(-0.02, -0.01), Interval(-0.020264113135339625, -0.00971833668405307), -0.015, 617, Interval(0.05, 0.2), Interval(0.0430899074525257, 0.20370378141862044), 0.125, 32, -0.2, 0.2, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.35, -0.34), Interval(-0.35019191648979947, -0.3396727621206105), -0.345, 467, Interval(-0.33, -0.331), Interval(-0.3310318637148097, -0.3299839974693975), -0.3305, 4353, -0.2, 0.2, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.35, -0.34), Interval(-0.3501386176639125, -0.3396946493234875), -0.345, 457, Interval(-0.33, -0.331), Interval(-0.33103551986472873, -0.3299853536909749), -0.3305, 4277, -0.225, 0.225, 0.97),
]

# Passband INTERIOR leaks: same windows/passbands but the order is bumped above what the stored
# passband was resolved for, so the now-sharper peak drops below its cutoff INSIDE the (fixed, wider)
# passband. Kept OUT of all_bad / all_filters so the "passband interiors are always clean" invariant
# is not contradicted; used only to exercise validate_passbands' false branch.
bad_passband = [
    Polfed.DoubleDeltaFilter(Interval(-0.12, -0.01), Interval(-0.12229514949186684, 0.010190313131929547), -0.065, 40, Interval(0.05, 0.2), Interval(0.038272886028551986, 0.20252928217256247), 0.125, 40, -0.2, 0.35, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.4, -0.19999999999999998), Interval(-0.4023261311339191, -0.19177847413648613), -0.3, 25, Interval(0.19, 0.21000000000000002), Interval(0.1891845315197194, 0.2102943063542795), 0.2, 275, -0.25, 0.175, 0.97),
    Polfed.DoubleDeltaFilter(Interval(-0.5, -0.4), Interval(-0.5008587072317201, -0.396826570116449), -0.45, 55, Interval(0.95, 0.975), Interval(0.9464193913942973, 0.9753568545574903), 0.9624999999999999, 46, -0.245, 0.225, 0.97),
]

const all_bad     = vcat(bad_left, bad_valley, bad_right, bad_left_valley, bad_left_right, bad_valley_right, bad_all)
const all_filters = vcat(good_filters, all_bad)

# --- call each region check of validate.jl with the filter's own defaults ---------------------
const _ε    = Polfed.Defaults.filters.double_delta.Δε
const _rtol = Polfed.Defaults.filters.double_delta.leak_rtol
_P(f)            = Polfed._scalar_filter(f)
_passbands_ok(f) = Polfed.validate_passbands(_P(f), f, _ε, _rtol, Polfed.Defaults.filters.double_delta.npts_within_passband(f.order1, f.order2))
_tails_ok(f)     = Polfed.validate_outer_tails(_P(f), f, _ε, _rtol, Polfed.Defaults.filters.double_delta.npts_on_sides_of_the_passband(f.order1, f.order2))
_valley_ok(f)    = Polfed.validate_between_passbands(_P(f), f, _ε, _rtol, Polfed.Defaults.filters.double_delta.npts_between_passbands(f.order1, f.order2))


@testset "DoubleDeltaFilter - validate filter" begin

    # ------------------------------------------------------------------------------------------
    # Integration: validate_filter(f) end-to-end — clean everywhere -> true, any leak -> false
    # ------------------------------------------------------------------------------------------
    @testset "validate_filter" begin
        @testset "good filters -> true (#$i)" for (i, f) in enumerate(good_filters)
            @test Polfed.validate_filter(f)
        end
        @testset "leaky filters -> false (#$i)" for (i, f) in enumerate(all_bad)
            @test Polfed.validate_filter(f) == false
        end
        @testset "passband-leak filters -> false (#$i)" for (i, f) in enumerate(bad_passband)
            @test Polfed.validate_filter(f) == false
        end
    end

    # ------------------------------------------------------------------------------------------
    # Passband interiors: clean for EVERY tail/valley-bucketed filter (good and bad), and the
    # bad_passband bucket (order bumped past its passband) is the one place they leak.
    # ------------------------------------------------------------------------------------------
    @testset "validate_passbands" begin
        @testset "interiors clean -> true (#$i)" for (i, f) in enumerate(all_filters)
            @test _passbands_ok(f)
        end
        @testset "interior leaks -> false (#$i)" for (i, f) in enumerate(bad_passband)
            @test _passbands_ok(f) == false
        end
    end

    # ------------------------------------------------------------------------------------------
    # Outer tails: validate_outer_tails = (left clean) && (right clean).
    #   true  iff BOTH tails clean:  good_filters, bad_valley
    #   false iff EITHER tail leaks: bad_left, bad_right, bad_left_valley, bad_left_right,
    #                                bad_valley_right, bad_all
    # ------------------------------------------------------------------------------------------
    @testset "validate_outer_tails" begin
        @testset "both tails clean -> true (#$i)" for (i, f) in enumerate(vcat(good_filters, bad_valley))
            @test _tails_ok(f)
        end
        @testset "a tail leaks -> false (#$i)" for (i, f) in
                enumerate(vcat(bad_left, bad_right, bad_left_valley, bad_left_right, bad_valley_right, bad_all))
            @test _tails_ok(f) == false
        end
    end

    # ------------------------------------------------------------------------------------------
    # Valley between the passbands.
    #   true  iff valley clean: good_filters, bad_left, bad_right, bad_left_right
    #   false iff valley leaks: bad_valley, bad_left_valley, bad_valley_right, bad_all
    # ------------------------------------------------------------------------------------------
    @testset "validate_between_passbands" begin
        @testset "valley clean -> true (#$i)" for (i, f) in
                enumerate(vcat(good_filters, bad_left, bad_right, bad_left_right))
            @test _valley_ok(f)
        end
        @testset "valley leaks -> false (#$i)" for (i, f) in
                enumerate(vcat(bad_valley, bad_left_valley, bad_valley_right, bad_all))
            @test _valley_ok(f) == false
        end
    end

    # ------------------------------------------------------------------------------------------
    # Guard: passbands must lie inside the spectrum  (validate_passbands_within_spectrum)
    # ------------------------------------------------------------------------------------------
    @testset "validate_passbands_within_spectrum" begin
        f = good_filters[3]                              # a comfortably-interior filter

        @testset "passbands inside [-1,1] -> true, no throw" begin
            @test Polfed.validate_passbands_within_spectrum(f, _ε)
        end

        @testset "passband1 past the left edge -> throws" begin
            g = deepcopy(f); g.passband1 = Interval(-1.2, g.passband1.right)
            @test_throws ErrorException Polfed.validate_passbands_within_spectrum(g, _ε)
            @test_throws ErrorException Polfed.validate_filter(g)
        end

        @testset "passband2 past the right edge -> throws" begin
            g = deepcopy(f); g.passband2 = Interval(g.passband2.left, 1.2)
            @test_throws ErrorException Polfed.validate_passbands_within_spectrum(g, _ε)
            @test_throws ErrorException Polfed.validate_filter(g)
        end
    end

    # ------------------------------------------------------------------------------------------
    # Primitive: two-sided band check  (_within_band). Analytic P = identity on [-0.5, 0.5].
    # ------------------------------------------------------------------------------------------
    @testset "_within_band" begin
        P = x -> x

        @testset "P inside [lower, upper] -> true" begin
            @test Polfed._within_band(P, -0.5, 0.5, -1.0, 1.0; npts = 200, rtol = 0.0)
        end

        @testset "P exceeds upper -> false" begin
            @test Polfed._within_band(P, -0.5, 0.5, -1.0, 0.2; npts = 200, rtol = 0.0) == false
        end

        @testset "P below lower -> false" begin
            @test Polfed._within_band(P, -0.5, 0.5, -0.2, 1.0; npts = 200, rtol = 0.0) == false
        end

        @testset "rtol relaxes the bound" begin
            Q = x -> 0.21                                # constant just above upper = 0.2
            @test Polfed._within_band(Q, -0.5, 0.5, -1.0, 0.2; npts = 50, rtol = 0.0)  == false
            @test Polfed._within_band(Q, -0.5, 0.5, -1.0, 0.2; npts = 50, rtol = 0.1)          # 0.2·1.1 = 0.22 > 0.21
        end
    end

end
