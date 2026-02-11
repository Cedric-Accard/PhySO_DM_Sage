from parser import Parser

p = Parser("NIHAO_runb_hydro_0_1.000000_nspe2_nclass1_bs2000/SR.log", verbose=True)

# best_overall = p.get_most_rewarded(5)
best_physical = p.get_physical_expr(20)

print(best_physical)
