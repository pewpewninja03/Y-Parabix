import csv, sys, subprocess, filecmp;

def perf_stat_counts(filename, perf_counters):
    src = open(filename)
    # The following are names for the observed output fields with pert stat -x, -r
    # These are described in the man page for pert-stat, but in a slightly 
    # different order.
    reader = csv.DictReader(src, ['count', 'units', 'event', 'variance', 'runtime', 'pct', 'aggregate', 'aggregate units'])
    result_dict = {}
    for row in reader:
        if row['event'] in perf_counters:
            result_dict[row['event']] = row['count']
    return result_dict

def run_with_perf_stat(program_under_test, args, perf_counters):
    # Now do a perf stat run with a repeat count of 5 and generating csv output
    statsf = open("/tmp/perfout", "w")
    perf_args = ["perf", "stat", "-x,", "-r5", program_under_test] + args
    subprocess.run(perf_args, encoding="utf-8", stdout=subprocess.DEVNULL, stderr=statsf)
    statsf.close()
    return perf_stat_counts("/tmp/perfout", perf_counters)

class PerformanceTester:
    def __init__(self, program_under_test, fixed_flags = []):
        self.PUT = program_under_test
        self.fixed_flags = fixed_flags
        self.functional_keys =[]
        self.performance_parms =[]
        self.positional_parameter_list = []
        self.visible_parameters = []
        self.performance_counters = ['instructions', 'cycles', 'branches', 'branch-misses']
        self.parameter_map = {}

    def addKFunctionalKey(self, keyword, choices):
        self.functional_keys.append(keyword)
        self.parameter_map[keyword] = choices

    def addPerformanceKey(self, keyword, choices):
        self.performance_parms.append(keyword)
        self.parameter_map[keyword] = choices

    def addPositionalParameter(self, name, choices):
        self.positional_parameter_list.append(name)
        self.parameter_map[name] = choices

    def combinations(self, parameter_list):
        if parameter_list == []:
            return [{}]
        else:
            tail_combos = self.combinations(parameter_list[1:])
            combos = []
            p = parameter_list[0]
            for choice in self.parameter_map[p]:
                for combo in tail_combos:
                    new_combo = {k : combo[k] for k in combo.keys()}
                    new_combo[p] = choice
                    combos.append(new_combo)
            return combos

    def run_tests(self, report_file):
        functional_parms = self.positional_parameter_list + self.functional_keys
        all_parms = functional_parms + self.performance_parms
        self.visible_parameters = [p for p in all_parms if len(self.parameter_map[p]) > 1]
        self.performance_combos = self.combinations(self.performance_parms)
        csv_sink = open(report_file, 'w')
        self.writer = csv.DictWriter(csv_sink, self.visible_parameters + self.performance_counters)
        self.writer.writeheader()
        functional_combos = self.combinations(functional_parms)
        for combo_map in functional_combos:
            self.run_functional_combo(combo_map)

    def run_functional_combo(self, combo_map):
        positional_parms = [combo_map[p] for p in self.positional_parameter_list]
        keyword_parms = [kw + "=" + combo_map[kw] for kw in self.functional_keys]
        functional_parms = self.fixed_flags + positional_parms + keyword_parms
        # First run the PUT with this functional arg combination and no
        # performance parameters to determine the expected output for all
        # combinations with these functional parameters and the performance
        # parameter combinations
        expectedf = open("/tmp/expected_out", "w")
        expected_msgf = open("/tmp/expected_msg", "w")
        subprocess.run([self.PUT] + functional_parms, encoding="utf-8", stdout=expectedf, stderr=expected_msgf)
        expectedf.close()
        expected_msgf.close()
        for performance_parm_map in self.performance_combos:
            performance_parms = [kw + "=" + performance_parm_map[kw] for kw in self.performance_parms]
            # run once with the full set of parameters to both create the cached executable
            # and check for correct output.
            outf = open("/tmp/test_output", "w")
            msgf = open("/tmp/test_msg", "w")
            parms = functional_parms + performance_parms
            subprocess.run([self.PUT] + parms, encoding="utf-8", stdout=outf, stderr=msgf)
            outf.close()
            msgf.close()
            if filecmp.cmp("/tmp/expected_out", "/tmp/test_output"):
                result_map = run_with_perf_stat(self.PUT, parms, self.performance_counters)
                for p in self.visible_parameters:
                    if p in self.performance_parms:
                        result_map[p] = performance_parm_map[p]
                    else:
                        result_map[p] = combo_map[p]
                self.writer.writerow(result_map)
            else:
                print("Test output does not match expected output", file=sys.stderr)
                print("  functional parms:", self.PUT, functional_parms)
                print("  performance parms:", performance_parms)
