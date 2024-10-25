let register info =
  let constructors =
    [
      ("Real", 2);
      ("Population", 2);
      ("Pred", 2);
      ("Conj", 2);
      ("Disj", 2);
      ("Impl", 2);
      ("Equiv", 2);
      ("StatTau", 2);
      ("StatB", 2);
      ("NormalD", 2);
    ]
  in
  List.iter
    (fun (s, i) -> Hashtbl.add info.Cameleer.Odecl.info_arith_construct s i)
    constructors
