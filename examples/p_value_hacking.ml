open CameleerBHL

module P_value_hacking = struct
  open Ttest

  (* Distributions and formulas *)
  let ppl_new = NormalD (Param "mu", Param "var")
  let h_new_l = mean ppl_new $< const_term 1.0
  let h_new_u = mean ppl_new $> const_term 1.0
  let h_new = mean ppl_new $!= const_term 1.0


  (* Example of p-value hacking *)
  (* This program is INCORRECT and so its verification FAILS *)
  let ex_hack trial1 trial2 =
    let p1 = exec_ttest_1samp ppl_new 1.0 trial1 Two in
    let p2 = exec_ttest_1samp ppl_new 1.0 trial2 Two in
    let p = min p1 p2 in (* This is INCORRECT *)
    (p1, p2, p)
  (*@ (p1, p2, p) = ex_hack trial1 trial2
    requires
      is_empty (!st) /\ sampled trial1 ppl_new /\ sampled trial2 ppl_new /\
      (World (!st) interp) |= Possible h_new_l /\ (World (!st) interp) |= Possible h_new_u
    ensures
      (Leq p = compose_pvs h_new !st
      && (World !st interp) |= StatB (Leq p) h_new)
   *)


  (* This is CORRECT *)
  let ex_correct trial1 trial2 =
    let p1 = exec_ttest_1samp ppl_new 1.0 trial1 Two in
    let p2 = exec_ttest_1samp ppl_new 1.0 trial2 Two in
    let p = p1 +. p2 in (* This is CORRECT *)
    (p1, p2, p)
  (*@ (p1, p2, p) = ex_hack trial1 trial2
    requires
      is_empty (!st) /\ sampled trial1 ppl_new /\ sampled trial2 ppl_new /\
      (World (!st) interp) |= Possible h_new_l /\ (World (!st) interp) |= Possible h_new_u
    ensures
      (Leq p = compose_pvs h_new !st
      && (World !st interp) |= StatB (Leq p) h_new)
   *)
end
