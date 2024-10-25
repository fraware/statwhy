# StatWhy

This artifact comprises StatWhy, a software tool for automatically verifying the correctness of statistical hypothesis testing programs.
In this file, we first present how to install the StatWhy tool. We then show how to verify the programs that implement the hypothesis testing examples addressed in our paper.


## Before Installing

The submitted zip archive must be copied into the VM (e.g., via a shared folder).
During installation, internet access is required to download external softwares, such as OCaml and CVC5.
It is recommended to update the system before installation via Software Updater.


## Installation

Unzip the source code and move to the root directory:
```bash
unzip statwhy-1.1.0.zip
cd statwhy-1.1.0
```

Run the following command:
```bash
source install.sh
```
During installation, you will be prompted to enter `y` or `n` multiple times.
Press `y` when prompted.

After installation, restart the machine or log in again to apply the changes made to `~/.profile`.


## Executing examples

Move to the `examples` directory:
``` bash
cd examples
```

As with Example 1, we provide a detailed explanation of how to use StatWhy.
You can run the other examples in a similar way to Example 1;
therefore we omit the details of such instructions for Example 2 and later.


### Example 1: one-sample t-test (Section 6.1 in our paper)

In this example, we demonstrate how to verify a program that conducts a one-sample t-test (shown in Section 6.1 in our paper), which is used to compare a population mean with a constant.
To verify the OCaml program `example1.ml`, run the following command:
``` bash
statwhy example1.ml 
``` 
This will launch the Why3 IDE as follows:

![Screenshot of the Why3 IDE](./screenshots/example1-why3-ide.png?raw=true "The Why3 IDE screen.")

There are two verification conditions (VCs) to be discharged: example1'vc (VC for example1) and example1''vc (VC for example1').
Right-click on the first goal and select 'StatWhy' (**Not 'CVC5 1.2.0' or the other items**), or press '4' after selecting the goal to make StatWhy try to discharge the goal.
If the prover successfully verifies the goal, a check mark will appear next to it as follows.

![Screenshot of the Why3 IDE showing a successfully discharged VC](./screenshots/example1-successful.png?raw=true "The Why3 IDE screen successfully discharged example1'vc.")

The second goal, example1''vc, is similar to the first, but lacks one of the preconditions: `sampled d t_n`.
Since this precondition for the t-test is missing, it is not appropriate to use the t-test.
In fact, if you press '4' on example1'vc, **StatWhy will fail to discharge the goal**, as shown below.
At this point, StatWhy generates sub-goals by applying transformations and attempts to prove them.
By examining the results of the sub-goals, you can see which conditions are missing in the annotation of the program.

![Screenshot of the Why3 IDE showing a VC that could not be discharged](./screenshots/example1-timed-out.png?raw=true "The Why3 IDE screen showing a goal that StatWhy could not discharge.")

**We emphasize that the above screenshot is the intended result**, in which several verification conditions are not discharged due to the lack of the precondition.
In the above screenshot, the condition `ppl @ d = NormalD ...` fails to discharge, which corresponds to the definition of the missing precondition `sampled d t_n`.


### Example 2: paired t-test (Section 6.2 in our paper)

In this example, we show how to verify a program that conducts a paired t-test (shown in Section 6.2 in our paper), which is used when there is a pairing or matching between two datasets.

Run the following command:
``` bash
statwhy example2.ml
```
The goal, `example2'vc`, should be successfully discharged.
For details, see Section 6.2 of the User Documentation, in the subsection 'Paired t-test vs. Non-Paired t-test.'


### Example 3: t-tests with equal vs. unequal variances (Section 6.2 in our paper)

In this example, we demonstrate how to verify a program that conducts t-tests for the comparison of two population means (shown in Section 6.2 in our paper).

Run the following command to verify `example3.ml`:
``` bash
statwhy example3.ml
```
You should see the goals, `example3_eq'vc` and `example3_neq'vc`, discharged successfully.
For details, refer to Section 6.2 of the User Documentation, in the subsection 'Equal vs. Unequal Variance.'


### Example 4: disjunctive vs. conjunctive hypotheses (Section 6.3 in our paper)

In this example, we demonstrate how to verify a program that covers the verification of combined tests (shown in Section 6.3 in our paper).

Run the following command:
``` bash
statwhy example4.ml
```

These four VCs cannot be immediately discharged, but StatWhy automatically splits these VCs into smaller ones and applies computations and simplifications to them.
Finally, these sub-goals will be discharged by the prover.

For details, refer to Section 6.3 of the User Documentation.

### Example 5: p-value hacking (Section 6.3 in our paper)

In this example (shown in Section 6.3 in our paper), we show that StatWhy can automatically check whether the p-values are correctly calculated and prevent p-value hacking.
Run the following command:
``` bash
statwhy example5.ml
```

**The VC, example5'vc, cannot be discharged**; StatWhy points out the p-value hacking in the code, which reports a lower p-value than the actual one.
This is implied by the fact that `Eq p = compose_pvs fmlA !st` is reduced to `false` after applying several transformations:

![Screenshot of the Why3 IDE showing the specific condition that failed to discharge](./screenshots/example5-incorrect-pvalue.png?raw=true "The Why3 IDE screen showing the specific condition that failed to discharge.")

On the other hand, the correct calculation of p-value, `Leq (p1 +. p2) = compose_pvs fmlA!st`, is successfully discharged:

![Screenshot of the Why3 IDE highlighting the correct p-value calculation](./screenshots/example5-correct-pvalue.png?raw=true "The Why3 IDE screen where the correct calculation of p-value is highlighted.")

For details, see Section 6.3 of the User Documentation, specifically in the subsection 'p-value hacking.'


### Example 6: Tukey's HSD test (Section 6.4 in our paper)

This example demonstrates the verification of a multiple comparison method called Tukey's HSD test.

Run the following command:
``` bash
statwhy example6.ml
```

These two VCs should be successfully discharged after applying of several transformations.

See Section 6.5 of the User Documentation, in the subsection "Tukey's HSD Test" for details.
