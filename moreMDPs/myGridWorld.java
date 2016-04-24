// basic
import burlap.oomdp.auxiliary.DomainGenerator;
import burlap.oomdp.core.*;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.common.UniformCostRF;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.QValue;
import java.util.*;
// domain
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
// algorithms
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
// visualization
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.visualizer.Visualizer;

/**
 * Grid World domain.
 *
 * use int[][] to represent initial grid. 1 means wall. 
 *
 */
public class myGridWorld {	
	
    GridWorldDomain dg;
    Domain domain;
    State initState;
    RewardFunction rf;
    TerminalFunction tf;
    SimpleHashableStateFactory hashFactory;

    private int MAX_ITERATIONS = 300;
    private int INCREMENT = 20;
    private int MAX_ITERATIONS_QL = MAX_ITERATIONS;
    private int INCREMENT_QL = INCREMENT;

    private List<Integer> numOfIters = new ArrayList<Integer>();

    private List<Integer> VISteps = new ArrayList<Integer>();
    private List<Integer> PISteps = new ArrayList<Integer>();
    private List<Integer> QLSteps = new ArrayList<Integer>();
	
    private List<Integer> VITime = new ArrayList<Integer>();
    private List<Integer> PITime = new ArrayList<Integer>();
    private List<Integer> QLTime = new ArrayList<Integer>();

    private List<Double> VIReward = new ArrayList<Double>();
    private List<Double> PIReward = new ArrayList<Double>();
    private List<Double> QLReward = new ArrayList<Double>();
   
    public myGridWorld() {
        // -------------------------------------------------------------
        // 11 by 11 with four walls by default construction

        //this.dg = new GridWorldDomain(11, 11);
        // this will set the size to 11 by 11 and set four walls
        //this.dg.setMapToFourRooms();
        //this.tf = new GridWorldTerminalFunction(10, 10);

        // -------------------------------------------------------------
        // 5 by 5 grid
        
        this.dg = new GridWorldDomain(
                                      new int[][] { 
                                          { 0, 0, 1, 0, 0},
                                          { 0, 0, 0, 0, 0},
                                          { 1, 0, 1, 0, 1},
                                          { 0, 0, 0, 0, 0},
                                          { 0, 0, 1, 0, 0} }
                                      );
        this.tf = new GridWorldTerminalFunction(4, 4);
        
        // -------------------------------------------------------------
        // 10 by 10 grid
        /*
        this.dg = new GridWorldDomain(
                                      new int[][] { 
                                          {0,0,0,0,1,1,0,0,0,0},
                                          {0,0,0,0,1,1,0,0,0,0},
                                          {0,0,0,0,0,0,0,0,0,0},
                                          {0,0,0,0,0,0,0,0,0,0},
                                          {1,1,0,0,1,1,0,0,1,1},
                                          {1,1,0,0,1,1,0,0,1,1},
                                          {0,0,0,0,0,0,0,0,0,0},
                                          {0,0,0,0,0,0,0,0,0,0},
                                          {0,0,0,0,1,1,0,0,0,0},
                                          {0,0,0,0,1,1,0,0,0,0} }
                                      );
        this.tf = new GridWorldTerminalFunction(9, 9);
        */
        // -------------------------------------------------------------
        // 15 by 15 grid
        /*
        this.dg = new GridWorldDomain(
                                      new int[][] { 
                                          {0,0,0,0,0,0,1,1,1,0,0,0,0,0,0},
                                          {0,0,0,0,0,0,1,1,1,0,0,0,0,0,0},
                                          {0,0,0,0,0,0,1,1,1,0,0,0,0,0,0},
                                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                          {1,1,1,0,0,0,1,1,1,0,0,0,1,1,1},
                                          {1,1,1,0,0,0,1,1,1,0,0,0,1,1,1},
                                          {1,1,1,0,0,0,1,1,1,0,0,0,1,1,1},
                                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                          {0,0,0,0,0,0,1,1,1,0,0,0,0,0,0},
                                          {0,0,0,0,0,0,1,1,1,0,0,0,0,0,0},
                                          {0,0,0,0,0,0,1,1,1,0,0,0,0,0,0} }
                                      );
        this.tf = new GridWorldTerminalFunction(14, 14);
        */
        // -------------------------------------------------------------
        
        // move to intended direction with p=0.8, and p=0.2 to other directions
	this.dg.setProbSucceedTransitionDynamics(0.8);	

        this.domain = this.dg.generateDomain();

        this.initState = GridWorldDomain.getOneAgentNoLocationState(this.domain);
        GridWorldDomain.setAgent(this.initState, 0, 0);

        this.rf = new UniformCostRF(); // -1 everywhere        
        this.hashFactory = new SimpleHashableStateFactory();

        //Visualizer v = GridWorldVisualizer.getVisualizer(this.dg.getMap());
        //VisualExplorer exp = new VisualExplorer(this.domain, v, this.initState);
        //exp.initGUI();
    }   

    /**
     * run the value iteration
     */
    public void computeValue(double gamma) {
    	double maxDelta = -1;

        for(int maxIterations = INCREMENT;
            maxIterations <= MAX_ITERATIONS; 
            maxIterations += INCREMENT ) {
            long startTime = System.nanoTime();
            ValueIteration vi = new ValueIteration(this.domain, 
                                                   this.rf, 
                                                   this.tf, 
                                                   gamma, 
                                                   this.hashFactory, 
                                                   maxDelta, 
                                                   maxIterations);
            vi.toggleDebugPrinting(false);
            Policy p = vi.planFromState(this.initState);
            VITime.add((int) (System.nanoTime()-startTime)/1000000);
            numOfIters.add(maxIterations);

            EpisodeAnalysis ea = p.evaluateBehavior(this.initState, 
                                                    this.rf, 
                                                    this.tf);
            VISteps.add(ea.numTimeSteps());
            VIReward.add(totalReward(ea));
        }
        System.out.println("----------------------------------------------------------");
        System.out.println("Number of iterations: ");
        for (int iter : numOfIters) {
            System.out.print(iter);
            System.out.print(",");
        }
        System.out.println();

        System.out.println("Time (ms): ");
        for (int t : VITime) {
            System.out.print(t);
            System.out.print(",");
        }
        System.out.println();

        System.out.println("Number of steps: ");
        for (int s : VISteps) {
            System.out.print(s);
            System.out.print(",");
        }
        System.out.println();

        System.out.println("Total reward: ");
        for (double r : VIReward) {
            System.out.print(r);
            System.out.print(",");
        }
        System.out.println();
        
    }
    
    /**
     * run the policy iteration
     */
    public void computePolicy(double gamma) {
    	double maxDelta = -1;
    	int maxEvaluationIterations = 1; // number of VI evaluation

        for(int maxIterations = INCREMENT;
            maxIterations <= MAX_ITERATIONS; 
            maxIterations += INCREMENT ) {  
            //long startTime = System.nanoTime();   
            long startTime = System.currentTimeMillis();
            PolicyIteration pi = new PolicyIteration(this.domain, 
                                                     this.rf, 
                                                     this.tf, 
                                                     gamma, 
                                                     this.hashFactory, 
                                                     maxDelta, 
                                                     maxEvaluationIterations,
                                                     maxIterations);            
            pi.toggleDebugPrinting(false);
            Policy p = pi.planFromState(this.initState);                
            //PITime.add((int) (System.nanoTime()-startTime)/1000000);            
            PITime.add((int) (System.currentTimeMillis()-startTime));            

            EpisodeAnalysis ea = p.evaluateBehavior(this.initState, 
                                                    this.rf, 
                                                    this.tf);
            PISteps.add(ea.numTimeSteps());
            PIReward.add(totalReward(ea));
        }

	System.out.println("----------------------------------------------------------");
        System.out.println("Time (ms): ");
        for (int t : PITime) {
            System.out.print(t);
            System.out.print(",");
        }
        System.out.println();

        System.out.println("Number of steps: ");
        for (int s : PISteps) {
            System.out.print(s);
            System.out.print(",");
        }
        System.out.println();

        System.out.println("Total reward: ");
        for (double r : PIReward) {
            System.out.print(r);
            System.out.print(",");
        }
        System.out.println();
    }
    
    /**
     * run Q learning
     */
    public void doQLearning(double gamma) {
        double qInit = 0.1; // initial Q values everywhere
        double learningRate = 0.1;        
        //int maxEpisodeSize = 1000;        

        EpisodeAnalysis ea = null;
        SimulatedEnvironment env = new SimulatedEnvironment(this.domain,
                                                            this.rf, 
                                                            this.tf, 
                                                            this.initState);
        numOfIters.clear();
        for(int maxIterations = INCREMENT_QL;
            maxIterations <= MAX_ITERATIONS_QL; 
            maxIterations += INCREMENT_QL ) 
        {  
            long startTime = System.nanoTime();   
            QLearning ql = new QLearning(this.domain,
                                         gamma,
                                         this.hashFactory,
                                         qInit, 
                                         learningRate);            
            ql.setLearningPolicy(new EpsilonGreedy(ql, 0.1));
            
            for (int i = 0; i < maxIterations; i++) {
                ea = ql.runLearningEpisode(env);
                env.resetEnvironment();
            }

            ql.initializeForPlanning(this.rf, this.tf, 1); // numEpisodesForPlanning = 1
            Policy p = ql.planFromState(this.initState);   
             
            QLTime.add((int) (System.nanoTime()-startTime)/1000000);
            
            numOfIters.add(maxIterations);
            
            QLSteps.add(ea.numTimeSteps());
            QLReward.add(totalReward(ea));
        }
        System.out.println("----------------------------------------------------------");
        System.out.println("Number of iterations: ");
        for (int iter : numOfIters) {
            System.out.print(iter);
            System.out.print(",");
        }
        System.out.println();

        System.out.println("Time (ms): ");
        for (int t : QLTime) {
            System.out.print(t);
            System.out.print(",");
        }
        System.out.println();

        System.out.println("Number of steps: ");
        for (int s : QLSteps) {
            System.out.print(s);
            System.out.print(",");
        }
        System.out.println();

        System.out.println("Total reward: ");
        for (double r : QLReward) {
            System.out.print(r);
            System.out.print(",");
        }
        System.out.println();
    }    

    /**
     * Calculate the total reward in one episode
     */
    public double totalReward(EpisodeAnalysis ea) {
        double totalReward = 0;
        
        for (int i = 0; i < ea.rewardSequence.size(); i++) {
            totalReward += ea.rewardSequence.get(i);
        }
        return totalReward;
    }	
    
    public void analyze(double gamma) {
        /*
        System.out.println("Value iteration:");
        computeValue(gamma);  
        
        System.out.println("==========================================================");
        System.out.println("Policy iteration:");
        computePolicy(gamma);  
        */
        System.out.println("==========================================================");
        System.out.println("Q learning:");        
        doQLearning(gamma);
        
    }

    public static void main(String[] args) {        
        myGridWorld mdp = new myGridWorld();
	
        double gamma = 0.9;
        mdp.analyze(gamma);

        System.out.println("Done!");
    }
}
