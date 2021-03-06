// basic
import burlap.oomdp.auxiliary.DomainGenerator;
import burlap.oomdp.core.*;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.auxiliary.common.NullTermination;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.QValue;
import java.util.List;
// domain
import burlap.domain.singleagent.graphdefined.GraphDefinedDomain;
// algorithms
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
// visualization
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.oomdp.visualizer.Visualizer;

/**
 * This MDP has 6 nodes, the rewards and transitions are set by hand, so it's not
 * straightforward to change size.
 */
public class GraphMDP implements QFunction {
		
    DomainGenerator dg;
    Domain domain;
    State initState;
    RewardFunction rf;
    TerminalFunction tf;
    SimpleHashableStateFactory hashFactory;
    int numStates;

    public GraphMDP(double p1, double p2, double p3, double p4) {
        this.numStates = 6;
        this.dg = new GraphDefinedDomain(numStates);
        
        //actions from initial state 0
        ((GraphDefinedDomain) this.dg).setTransition(0, 0, 1, 1.);
        ((GraphDefinedDomain) this.dg).setTransition(0, 1, 2, 1.);
        ((GraphDefinedDomain) this.dg).setTransition(0, 2, 3, 1.);
        
        //transitions from action "a" outcome state
        ((GraphDefinedDomain) this.dg).setTransition(1, 0, 1, 1.);
        
        //transitions from action "b" outcome state
        ((GraphDefinedDomain) this.dg).setTransition(2, 0, 4, 1.);
        ((GraphDefinedDomain) this.dg).setTransition(4, 0, 2, 1.);

        //transitions from action "c" outcome state
        ((GraphDefinedDomain) this.dg).setTransition(3, 0, 5, 1.);
        ((GraphDefinedDomain) this.dg).setTransition(5, 0, 5, 1.);
        
        this.domain = this.dg.generateDomain();
        this.initState = GraphDefinedDomain.getState(this.domain,0);
        this.rf = new FourParamRF(p1,p2,p3,p4);
        this.tf = new NullTermination();
        this.hashFactory = new SimpleHashableStateFactory();
    }

    /**
     * Three dumb methods in order to implement the QFunction interface, for
     * EpsilonGreedy initialization
     */
    @Override
    public List<QValue> getQs(State s) {
        return null;
    }
    @Override
    public QValue getQ(State s, AbstractGroundedAction a) {
        return null;
    }
    @Override
    public double value(State s) {
        return 0;
    }

    /**
     * define the reward function
     */
    public static class FourParamRF implements RewardFunction {
        double p1;
        double p2;
        double p3;
        double p4;
		
        public FourParamRF(double p1, double p2, double p3, double p4) {
            this.p1 = p1;
            this.p2 = p2;
            this.p3 = p3;
            this.p4 = p4;
        }
		
        @Override
        public double reward(State s, GroundedAction a, State sprime) { 
            int sid = GraphDefinedDomain.getNodeId(s);
            double r;
            
            if( sid == 0 || sid == 3 ) { // initial state or c1
                r = 0;
            }
            else if( sid == 1 ) { // a
                r = this.p1;
            }
            else if( sid == 2 ) { // b1
                r = this.p2;
            }
            else if( sid == 4 ) { // b2
                r = this.p3;
            }
            else if( sid == 5 ) { // c2
                r = this.p4;
            }
            else {
                throw new RuntimeException("Unknown state: " + sid);
            }
            
            return r;
        }
    }

    /**
     * run the value iteration
     */
    public ValueIteration computeValue(double gamma) {
    	double maxDelta = 0.0001;
    	int maxIterations = 1000;
    	ValueIteration vi = new ValueIteration(this.domain, 
                                               this.rf, 
                                               this.tf, 
                                               gamma, 
                                               this.hashFactory, 
                                               maxDelta, 
                                               maxIterations);
    	vi.planFromState(this.initState);
    	return vi;
    }
    
    /**
     * run the policy iteration
     */
    public PolicyIteration computePolicy(double gamma) {
    	double maxDelta = 0.0001;
    	int maxEvaluationIterations = 1000; // maxEvaluationIterations is redundant here
    	int maxPolicyIterations = 1000;        
    	PolicyIteration pi = new PolicyIteration(this.domain, 
                                                 this.rf, 
                                                 this.tf, 
                                                 gamma, 
                                                 this.hashFactory, 
                                                 maxDelta, 
                                                 maxEvaluationIterations,
                                                 maxPolicyIterations);
    	pi.planFromState(this.initState);
    	return pi;
    }
    
    /**
     * run Q learning
     */
    public QLearning doQLearning(double gamma, Policy learningPolicy) {
        double qInit = 0.1; // initial Q values everywhere
        double learningRate = 0.1;        
        int maxEpisodeSize = 1000;
        QLearning ql = new QLearning(this.domain,
                                     gamma,
                                     this.hashFactory,
                                     qInit, 
                                     learningRate,                                     
                                     //learningPolicy,
                                     maxEpisodeSize);
        SimulatedEnvironment env = new SimulatedEnvironment(this.domain,
                                                            this.rf, 
                                                            this.tf, 
                                                            this.initState);
        EpisodeAnalysis ea;
        ea = ql.runLearningEpisode(env, 1000);
        System.out.println(ea.numTimeSteps());
        return ql;
    }    

    public void analyze(double gamma) {
        System.out.println("Value iteration:");
        ValueIteration vi = computeValue(gamma);  

        System.out.println("Policy iteration:");
        PolicyIteration pi = computePolicy(gamma);  

        System.out.println("Q learning:");
        Policy learningPolicy = new EpsilonGreedy(this, 0.1);
        doQLearning(gamma, learningPolicy);
    }

    public static void main(String[] args) {
        double p1 = 5.;
        double p2 = 6.;
        double p3 = 3.;
        double p4 = 7.;
        GraphMDP mdp = new GraphMDP(p1,p2,p3,p4);
	
        double gamma = 0.6;

        mdp.analyze(gamma);
        System.out.println("Done!");
    }
}
