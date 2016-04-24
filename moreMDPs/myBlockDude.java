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
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeTF;
import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
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
 * Block Dude domain.
 *
 * width of the map: maxx; height of the map: maxy
 * use int[][] to represent the map, 0 means empty, 1 means fixed block.
 * use setBlock() to set movable block.
 *
 * use -1 everywhere as reward, to minimize the steps required
 *
 */
public class myBlockDude {	
	
    BlockDude dg;
    Domain domain;
    State initState;
    RewardFunction rf;
    TerminalFunction tf;
    SimpleHashableStateFactory hashFactory;

    int maxx = 14; // 10, 12, 14
    int maxy = 5;

    private int MAX_ITERATIONS = 20;
    private int INCREMENT = 2;
    private int MAX_ITERATIONS_QL = 300;
    private int INCREMENT_QL = 30;

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
   
    public myBlockDude() {
        
        this.dg = new BlockDude(this.maxx, this.maxy);                                
        this.domain = this.dg.generateDomain();

        this.initState = getLevel0(this.domain);
                
        this.rf = new UniformCostRF(); // 0 everywhere  
        this.tf = new BlockDudeTF();
        this.hashFactory = new SimpleHashableStateFactory();

        Visualizer v = BlockDudeVisualizer.getVisualizer(this.maxx, 
                                                         this.maxy);
        VisualExplorer exp = new VisualExplorer(this.domain, v, this.initState);
        exp.initGUI();
	
    }   

    /**
     * run the value iteration
     */
    public void computeValue(double gamma) {
    	double maxDelta = -1;

        //for(int maxIterations = INCREMENT;
        //    maxIterations <= MAX_ITERATIONS; 
        //    maxIterations += INCREMENT ) {
        for(int maxIterations = MAX_ITERATIONS;
            maxIterations > 0; 
            maxIterations -= INCREMENT ) {
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
            VITime.add(0, (int) (System.nanoTime()-startTime)/1000000);
            numOfIters.add(0, maxIterations);

            EpisodeAnalysis ea = p.evaluateBehavior(this.initState, 
                                                    this.rf, 
                                                    this.tf);
            VISteps.add(0, ea.numTimeSteps());
            VIReward.add(0, totalReward(ea));
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
    	double maxDelta = 0.0001; //-1;
    	int maxEvaluationIterations = 1000; // number of VI evaluation

        //for(int maxIterations = INCREMENT;
        //    maxIterations <= MAX_ITERATIONS; 
        //    maxIterations += INCREMENT ) {
        for(int maxIterations = MAX_ITERATIONS;
            maxIterations > 0; 
            maxIterations -= INCREMENT ) {
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
            PITime.add(0, (int) (System.currentTimeMillis()-startTime));               

            EpisodeAnalysis ea = p.evaluateBehavior(this.initState, 
                                                    this.rf, 
                                                    this.tf);
            PISteps.add(0, ea.numTimeSteps());
            PIReward.add(0, totalReward(ea));
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
        //for(int maxIterations = INCREMENT_QL;
        //    maxIterations <= MAX_ITERATIONS_QL; 
        //    maxIterations += INCREMENT_QL ) 
        for(int maxIterations = MAX_ITERATIONS_QL;
            maxIterations > 0; 
            maxIterations -= INCREMENT_QL ) 
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
             
            QLTime.add(0, (int) (System.nanoTime()-startTime)/1000000);

            numOfIters.add(0, maxIterations);

            QLSteps.add(0, ea.numTimeSteps());
            QLReward.add(0, totalReward(ea));
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
    
    /**
     * Define my own level
     */
    public State getLevel0(Domain domain){

        int [][] map = new int[this.maxx][this.maxy];
        BlockDudeLevelConstructor.addFloor(map);
        
        map[this.maxx/2-1][1] = 1;
        map[this.maxx/2-1][2] = 1;                
                
        State s = BlockDude.getUninitializedState(domain, 1); // domain, num of blocks
        BlockDude.setAgent(s, this.maxx-2, 1, 1, false); // s, x, y, dir, holding
        BlockDude.setExit(s, 0, 1);
        BlockDude.setBlock(s, 0, this.maxx/2+1, 1); // s, i, x, y
        
        BlockDude.setBrickMap(s, map);

        return s;
    }

    /**
     * solve the MDP and do RL
     */
    public void analyze(double gamma) {
        System.out.println("Value iteration:");
        //computeValue(gamma);  
        
        System.out.println("==========================================================");
        System.out.println("Policy iteration:");
        //computePolicy(gamma);  

        System.out.println("==========================================================");
        System.out.println("Q learning:");        
        doQLearning(gamma);
        
    }

    public static void main(String[] args) {        
        myBlockDude mdp = new myBlockDude();
        
        double gamma = 0.9;
        mdp.analyze(gamma);

        System.out.println("Done!");
    }
}
