package de.unima.ki.anyburl.algorithm;

import java.util.ArrayList;
import java.util.Random;

import de.unima.ki.anyburl.Settings;
import de.unima.ki.anyburl.data.Triple;
import de.unima.ki.anyburl.data.TripleSet;
import de.unima.ki.anyburl.structure.Path;
import de.unima.ki.anyburl.structure.Rule;

/**
* This class is responsible for sampling grounded pathes.
*
*/
public class PathSampler {

	private TripleSet ts;
	private Random rand = new Random();
	
	
	public static void main(String[] args) {
			
	}
	
	
	public PathSampler(TripleSet ts) {
		this.ts = ts;
	}
	
	
	public Path samplePath(int steps, boolean cyclic) {
		return samplePath(steps, cyclic, null);
	}
	
	/**
	 * Samples a normal path which is the seconds half of a Y path where the given path is the first half of the Y.
	 * 
	 * @param path
	 * @return A normal path that corresponds to an AC1 rule.
	 */
	public Path sampleYPath(Path path) {
		return samplePath(2, false, null, null, path);
	}
	
	
	public Path samplePath(int steps, boolean cyclic, Triple chosenHeadTriple) {
		return samplePath(steps, cyclic, chosenHeadTriple, null, null);
	}
	
	public Path samplePath(int steps, boolean cyclic, Triple chosenHeadTriple, Rule ruleToBeExtended, Path path) {
		String[] nodes = new String[1 + steps * 2];
		char[] markers = new char[steps];
		ArrayList<Triple> chosenTriples;
		if (Settings.SINGLE_RELATIONS != null) {
			int rdice = this.rand.nextInt(Settings.SINGLE_RELATIONS.length);
			String singleRelation = Settings.SINGLE_RELATIONS[rdice];
			chosenTriples = ts.getTriplesByRelation(singleRelation);
			if (chosenTriples.size() == 0) {
				System.err.println("chosen a SINGLE_RELATION=" + singleRelation + " that is not instantiated in the training data");
				System.exit(0);	
			}
		}
		else {
			chosenTriples = ts.getTriples();
		}
		Triple triple = null;
		if (chosenHeadTriple == null) {
			int dice = this.rand.nextInt(chosenTriples.size());
			triple = chosenTriples.get(dice);
		}
		else triple = chosenHeadTriple;
		
		if (Settings.FORBIDDEN_RELATIONS != null && Settings.FORBIDDEN_RELATIONS_AS_SET.contains(triple.getRelation())) return null;

		// TODO hardcoded test to avoid reflexive relations in the head
		if (triple.getHead().equals(triple.getTail())) return null;
		double dice = this.rand.nextDouble();
		if (ruleToBeExtended != null) {
			if (ruleToBeExtended.isXRule()) dice = 1;
			if (ruleToBeExtended.isYRule()) dice = 0;
		}
		/*
		if (fromHeadToTail != 0) {
			if (fromHeadToTail == 1) dice = 0.25;
			else dice = 0.75;
		}
		*/
		if (dice < 0.5) {
			markers[0] = '+';
			nodes[0] = triple.getHead();
			nodes[1] = triple.getRelation();
			nodes[2] = triple.getTail();
		}
		else {
			markers[0] = '-';
			nodes[2] = triple.getHead();
			nodes[1] = triple.getRelation();
			nodes[0] = triple.getTail();
		}
		if (path != null) {
			markers[0] = path.getMarkerAt(0);
			nodes[0] = path.getNodeAt(0);
			nodes[1] = path.getNodeAt(1);
			nodes[2] = path.getNodeAt(2);
		}
		
		// add next hop
		int index = 1;
		while (index < steps) {
			if (this.rand.nextDouble() < 0.5) {
				ArrayList<Triple> candidateTriples = ts.getTriplesByHead(nodes[index*2]);
				if (candidateTriples.size() == 0) return null;
				Triple nextTriple;
				if (cyclic && index + 1 == steps) {
					ArrayList<Triple> cyclicCandidateTriples = new ArrayList<>();
					for (Triple t : candidateTriples) {
						if (t.getTail().equals(nodes[0])) cyclicCandidateTriples.add(t);
					}
					if (cyclicCandidateTriples.size() == 0) return null;
					nextTriple = cyclicCandidateTriples.get(this.rand.nextInt(cyclicCandidateTriples.size()));
				}
				else {
					nextTriple = candidateTriples.get(this.rand.nextInt(candidateTriples.size()));
				}
				nodes[index*2+1] = nextTriple.getRelation();
				nodes[index*2+2] = nextTriple.getTail();
				markers[index] = '+';
			}
			else {
				ArrayList<Triple> candidateTriples = ts.getTriplesByTail(nodes[index*2]);
				if (candidateTriples.size() == 0) return null;
				Triple nextTriple;
				if (cyclic && index + 1 == steps) {
					ArrayList<Triple> cyclicCandidateTriples = new ArrayList<>();
					for (Triple t : candidateTriples) {
						if (t.getHead().equals(nodes[0])) cyclicCandidateTriples.add(t);
					}
					if (cyclicCandidateTriples.size() == 0) return null;
					nextTriple = cyclicCandidateTriples.get(this.rand.nextInt(cyclicCandidateTriples.size()));
				}
				else {
					nextTriple = candidateTriples.get(this.rand.nextInt(candidateTriples.size()));
				}
				nodes[index*2+1] = nextTriple.getRelation();
				nodes[index*2+2] = nextTriple.getHead();
				markers[index] = '-';
			}
			index++;
		}
		Path p = new Path(nodes, markers);
		if (steps == 1) return p;
		if (!cyclic && p.isCyclic()) return null;
		return p;
	}
	

	
}
