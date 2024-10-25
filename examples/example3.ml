open CameleerBHL

module Example3 = struct
  open Ttest

  let t_n1 = NormalD (Param "mu1", Param "var1")
  let t_n2 = NormalD (Param "mu2", Param "var2")

  let fmlA_l : formula =
    Atom (Pred ("<", [ RealT (mean t_n1); RealT (mean t_n2) ]))

  let fmlA_u : formula =
    Atom (Pred (">", [ RealT (mean t_n1); RealT (mean t_n2) ]))

  let fmlA : formula =
    Atom (Pred ("=!", [ RealT (mean t_n1); RealT (mean t_n2) ]))

  (* Execute Student's t-test for two population means *)
  let example3_eq (d : (float list * float list) dataset) : float =
    exec_ttest_ind_eq t_n1 t_n2 d Two
  (*@
    p = example3_eq d
    requires
      let (d1, d2) = d in
      is_empty (!st) /\
      non_paired d1 d2 /\
      sampled d1 t_n1 /\ sampled d2 t_n2 /\
      (World (!st) interp) |= eq_variance t_n1 t_n2 /\
      (World (!st) interp) |= Possible fmlA_l /\
      (World (!st) interp) |= Possible fmlA_u
    ensures
      Eq p = compose_pvs fmlA !st &&
      (World !st interp) |= StatB (Eq p) fmlA
  *)

  (* Execute Welch's t-test for two population means *)
  let example3_neq (d : (float list * float list) dataset) : float =
    exec_ttest_ind_neq t_n1 t_n2 d Two
  (*@
    p = example3_neq d
    requires
      let (d1, d2) = d in
      is_empty (!st) /\
      non_paired d1 d2 /\
      sampled d1 t_n1 /\ sampled d2 t_n2 /\
      (World (!st) interp) |= Not (eq_variance t_n1 t_n2) /\
      (World (!st) interp) |= Possible fmlA_l /\
      (World (!st) interp) |= Possible fmlA_u
    ensures
      Eq p = compose_pvs fmlA !st &&
      (World !st interp) |= StatB (Eq p) fmlA
  *)
end
