open Why3

let () =
  Trans.register_transform "subst_all_statwhy" Subst.subst_all
    ~desc:"Substitute@ with@ all@ equalities@ between@ \
           a@ constant@ and@ a@ term."
