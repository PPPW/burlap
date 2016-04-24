// basic
import burlap.oomdp.auxiliary.DomainGenerator;
import burlap.oomdp.core.*;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.common.UniformCostRF;
import burlap.oomdp.singleagent.common.NullRewardFunction;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.QValue;
import java.util.*;
// domain
import burlap.domain.singleagent.blocksworld.BlocksWorld;
import burlap.domain.singleagent.blocksworld.BlocksWorldVisualizer;
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
 * This MDP has 6 nodes, the rewards and transitions are set by hand, so it's not
 * straightforward to change size.
 */
public class myBlocksWorld {	
	
    BlocksWorld dg;
    Domain domain;
    State initState;
    RewardFunction rf;
    TerminalFunction tf;
    SimpleHashableStateFactory hashFactory;

    private int MAX_ITERATIONS = 20;
    private int INCREMENT = 4;
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
   
    public myBlocksWorld() {
        
        this.dg = new BlocksWorld();                                
        this.domain = this.dg.generateDomain();

        this.initState = BlocksWorld.getNewState(domain, 3);
                
        //this.rf = new UniformCostRF(); // 0 everywhere  
        //this.tf = new BlockDudeTF();
        this.hashFactory = new SimpleHashableStateFactory();
        
        VisualExplorer exp = new VisualExplorer(this.domain, 
                                                BlocksWorldVisualizer.getVisualizer(24), 
                                                this.initState);
        exp.addKeyAction("s", "stack");
        exp.addKeyAction("u", "unstack");
        exp.initGUI();
	
    }   

    /**
     * run the value iteration
     */
    public void computeValue(double gamma) {
    	double maxDelta = 0.0001;
    	//int maxIterations = 100;
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
    	double maxDelta = 0.0001;
    	int maxEvaluationIterations = 1000; // maxEvaluationIterations is redundant here
    	//int maxIterations = 1000;  
        for(int maxIterations = INCREMENT;
            maxIterations <= MAX_ITERATIONS; 
            maxIterations += INCREMENT ) {  
            long startTime = System.nanoTime();   
            PolicyIteration pi = new PolicyIteration(this.domain, 
                                                     this.rf, 
                                                     this.tf, 
                                                     gamma, 
                                                     this.hashFactory, 
                                                     maxDelta, 
                                                     maxEvaluationIterations,
                                                     maxIterations);
            Policy p = pi.planFromState(this.initState);                
            PITime.add((int) (System.nanoTime()-startTime)/1000000);            

            EpisodeAnalysis ea = p.evaluateBehavior(this.initState, 
                                                    this.rf, 
                                                    this.tf);
            PISteps.add(ea.numTimeSteps());
            PIReward.add(totalReward(ea));
        }
	
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

        for(int maxIterations = INCREMENT;
            maxIterations <= MAX_ITERATIONS; 
            maxIterations += INCREMENT ) 
        {  
            long startTime = System.nanoTime();   
            QLearning ql = new QLearning(this.domain,
                                         gamma,
                                         this.hashFactory,
                                         qInit, 
                                         learningRate);            
            ql.setLearningPolicy(new EpsilonGreedy(ql, 0.2));
            
            for (int i = 0; i < maxIterations; i++) {
                ea = ql.runLearningEpisode(env);
                env.resetEnvironment();
            }

            ql.initializeForPlanning(this.rf, this.tf, 1); // numEpisodesForPlanning = 1
            Policy p = ql.planFromState(this.initState);   
             
            QLTime.add((int) (System.nanoTime()-startTime)/1000000);
            QLSteps.add(ea.numTimeSteps());
            QLReward.add(totalReward(ea));
        }

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

    /**
     * solve the MDP and do RL
     */
    public void analyze(double gamma) {
        System.out.println("Value iteration:");
        //computeValue(gamma);  
        
        //System.out.println("==========================================================");
        //System.out.println("Policy iteration:");
        //computePolicy(gamma);  

        //System.out.println("==========================================================");
        //System.out.println("Q learning:");        
        //doQLearning(gamma);
        
    }

    public static void main(String[] args) {        
        myBlocksWorld mdp = new myBlocksWorld();
        
        double gamma = 0.9;
        mdp.analyze(gamma);

        System.out.println("Done!");
    }
}
