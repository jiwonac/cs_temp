use clap::{App, Arg};
use egg::*;
use cs::language::*;
use std::time::{Duration, Instant};

fn main() {
    // Parse arguments
    let matches = App::new("Tamago")
    .arg(
        Arg::with_name("program")
            .short("program")
            .long("program")
            .takes_value(true)
            .help("Specify input program")
            .required(true),
    )
    .arg(
        Arg::with_name("n_iter")
            .long("n_iter")
            .takes_value(true)
            .default_value("100")
            .help("Max number of iterations for egg to run"),
    )
    .arg(
        Arg::with_name("n_sec")
            .long("n_sec")
            .takes_value(true)
            .default_value("10")
            .help("Max number of seconds for egg to run"),
    )
    .arg(
        Arg::with_name("n_nodes")
            .long("n_nodes")
            .takes_value(true)
            .default_value("100000")
            .help("Max number of nodes for egraph"),
    )
    .get_matches();

    optimize(matches);
}

type MyRunner = Runner<TnsrLang, TnsrAnalysis, ()>;

fn optimize(matches: clap::ArgMatches) {
    let program = matches.value_of("program").map(String::from).unwrap();
    let program_expr: RecExpr<TnsrLang> = program.parse().unwrap();
    let n_iter = matches.value_of("n_iter").unwrap().parse::<usize>().unwrap();
    let n_sec = Duration::new(matches.value_of("n_sec").unwrap().parse::<u64>().unwrap(), 0);
    let n_nodes = matches.value_of("n_nodes").unwrap().parse::<usize>().unwrap();

    let initial_expr: RecExpr<TnsrLang> = program.parse().unwrap();
    let initial_runner = MyRunner::new(Default::default()).with_expr(&initial_expr);
    let initial_tnsr_cost = TnsrCost {
        egraph: &initial_runner.egraph,
    };
    let initial_extractor = Extractor::new(&initial_runner.egraph, initial_tnsr_cost);
    let (initial_cost, _) = initial_extractor.find_best(initial_runner.roots[0]);
    println!("Initial cost: {}", initial_cost);

    let runner = MyRunner::new(Default::default())
                    .with_node_limit(n_nodes)
                    .with_time_limit(n_sec)
                    .with_iter_limit(n_iter)
                    .with_expr(&program_expr);
    
    let start_time = Instant::now();
    let runner = runner.run(&rules::<TnsrAnalysis>());
    let sat_duration = start_time.elapsed();
    let num_iter_sat = runner.iterations.len() - 1;

    println!("Runner complete!");
    println!("  Nodes: {}", runner.egraph.total_size());
    println!("  Classes: {}", runner.egraph.number_of_classes());
    println!("  Stopped: {:?}", runner.stop_reason.unwrap());
    println!("  Time taken: {:?}", sat_duration);
    println!("  Number of iterations: {:?}", num_iter_sat);

    let (_, _, avg_nodes_per_class, num_edges, num_programs) =
    get_stats(&runner.egraph);
    println!("  Average nodes per class: {}", avg_nodes_per_class);
    println!("  Number of edges: {}", num_edges);
    println!("  Number of programs: {}", num_programs);

    let (egraph, root) = (runner.egraph, runner.roots[0]);
    let tnsr_cost = TnsrCost {
        egraph: &egraph,
    };
    let start_time = Instant::now();
    let extractor = Extractor::new(&egraph, tnsr_cost);
    let (best_cost, best) = extractor.find_best(root);
    let duration = start_time.elapsed();

    println!("Post-saturation extractor complete!");
    println!("  Time taken: {:?}", duration);
    println!("  Best cost: {:?}", best_cost);

    println!("Extracted program:\n {}", best.pretty(40 as usize));

    let lp_tnsr_cost = TnsrCost {
        egraph: &egraph,
    };
    let mut lp_extractor = LpExtractor::new(&egraph, lp_tnsr_cost);
    let lp_start_time = Instant::now();
    let lp_best = lp_extractor.solve(root);
    let lp_duration = lp_start_time.elapsed();

    println!("ILP extraction complete!");
    println!("   Time taken: {:?}", lp_duration);
    //println!("   Best cost: {:?}", lp_best_cost);
    println!("Extracted program:\n {}", lp_best.pretty(40 as usize));
}

fn get_stats(egraph: &EGraph<TnsrLang, TnsrAnalysis>) -> (usize, usize, f32, usize, f32) {
    let num_enodes = egraph.total_size();
    let num_classes = egraph.number_of_classes();
    let avg_nodes_per_class = num_enodes as f32 / (num_classes as f32);
    let num_edges = egraph
        .classes()
        .fold(0, |acc, c| c.iter().fold(0, |sum, n| n.len() + sum) + acc);
    let num_programs = egraph
        .classes()
        .fold(0.0, |acc, c| acc + (c.len() as f32).log2());
    (
        num_enodes,
        num_classes,
        avg_nodes_per_class,
        num_edges,
        num_programs,
    )
}