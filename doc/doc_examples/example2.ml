open CameleerBHL

module Example2 = struct
  open Ttest

  let t_n1 = NormalD (Param "mu1", Param "var1")
  let t_n2 = NormalD (Param "mu2", Param "var2")

  let fmlA_l : formula =
    Atom (Pred ("<", [ RealT (mean t_n1); RealT (mean t_n2) ]))

  let fmlA_u : formula =
    Atom (Pred (">", [ RealT (mean t_n1); RealT (mean t_n2) ]))

  let fmlA : formula =
    Atom (Pred ("!=", [ RealT (mean t_n1); RealT (mean t_n2) ]))

  (* Execute Paired t-test for two population means *)
  let example2 (d : (float list * float list) dataset) : float =
    exec_ttest_paired t_n1 t_n2 d Two
  (*@
    p = example2 d
    requires
      let (d1, d2) = d in
      is_empty (!st) /\
      paired d1 d2 /\
      sampled d1 t_n1 /\ sampled d2 t_n2 /\
      (World (!st) interp) |= Possible fmlA_l /\
      (World (!st) interp) |= Possible fmlA_u
    ensures
      Eq p = compose_pvs fmlA !st &&
      (World !st interp) |= StatB (Eq p) fmlA
  *)
end
