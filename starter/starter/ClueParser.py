#!/usr/bin/env python
# CS124 Homework 5 Jeopardy
# Original written in Java by Sam Bowman (sbowman@stanford.edu)
# Ported to Python by Milind Ganjoo (mganjoo@stanford.edu)

from NaiveBayes import NaiveBayes
import sys
import re
import itertools as it

class ClueParser:
    def __init__(self):
        # TODO: if your implementation requires one or more trained classifiers (it probably does), you should declare it/them here.
        # Remember to import the class at the top of the file (from NaiveBayes import NaiveBayes)
        # e.g. self.classifier = NaiveBayes()
        self.classifer = NaiveBayes()
        pass

    def feature_extractor(self, clue):
        """Given a clue represented as a raw string of text, extract features of the clue and return them as a list or set."""
	features = []
	words = clue.split()
	for word in words:
		features.append(word)
	special_words = ["married", "college", "university", "president", "based", "born", "mayor", "located", "year", "died", "birthplace", "birth"]
	for word in special_words:
		if word in clue:
			features.append("1")
		else:	
			features.append("0")
	return features
	
    def train(self, clues, parsed_clues):
        """Trains the model on clues paired with gold standard parses."""
	featuresList = []
        relations = []
	for clue in clues:
		featuresList.append(self.feature_extractor(clue))
	for clue in parsed_clues:
		relations.append(clue[:clue.find(":")])
	self.classifer.addExamples(featuresList, relations)

    def getMatch(self, clue, pattern, relation):
	matches = re.findall(pattern, clue)
	specialWords = ["is", "was", "married", "is married", "president", "leader", "head", "founder", "husband", "marriage", "college", "university", "degree", "wife", "has been", "in charge of", "based", "headquartered", "headquared", "run", "located", "is in", "led", "headed"]
	words = []
        for m in matches:
		if isinstance(m, tuple):
        	        for index, elem in enumerate(m):
				if elem not in specialWords and len(elem) > 0:
                        		words.append(elem)
		else:
			return m;
	if len(words) > 1 and words[0] != words[1]:
		return words[0] + ", " + words[1]
	elif len(words) > 0:
		return words[0]
        return 'no match'

    def getEntity(self, relation, clue):
	pattern = ""
	if relation == "wife_of":
		pattern = 'married [\sa-zA-Z]*<PERSON>([\.A-Za-z\s\']*)</PERSON>|wife of [\sa-zA-Z]*<PERSON>([\.A-Za-z\s\']*)</PERSON>|<PERSON>([.A-Za-z\s]*)</PERSON> married|wife is [\sA-Za-z]*<PERSON>([\.\'A-Za-z\s]*)</PERSON>|<PERSON>([\.\'A-Za-z\s]*)</PERSON>\'s [\sA-Za-z]*(wife|marriage)|the wedding between her and [\sA-Za-z]*<PERSON>([\.\'A-Za-z\s]*)</PERSON>|<PERSON>([\.\'A-Za-z\s]*)</PERSON> (is married|married)'
	elif relation == "husband_of":
		pattern = 'married [\sa-zA-Z<>/]*<PERSON>([\.A-Za-z\s\']*)</PERSON>|husband of [\sa-zA-Z<>/]*<PERSON>([\.A-Za-z\s\']*)</PERSON>|<PERSON>([.A-Za-z\s]*)</PERSON> married|wife is [\sA-Za-z<>/]*<PERSON>([\.\'A-Za-z\s]*)</PERSON>|<PERSON>([\.\'A-Za-z\s]*)</PERSON>\'s [\sA-Za-z<>/]*(husband|marriage)|the wedding between him and [\sA-Za-z<>/]*<PERSON>([\.\'A-Za-z\s]*)</PERSON>|<PERSON>([\.\'A-Za-z\s]*)</PERSON> (is married|married)|<PERSON>([\.A-Za-z\s\']*)</PERSON> met this man'
	elif relation == "college_of":
		if clue.count("PERSON")>2:
			pattern = '<PERSON>([\.A-Za-z\s\']*)</PERSON>\'s alma mater|<PERSON>([\.A-Za-z\s\']*)</PERSON>[\sa-zA-Z]*(college|university|degree)|alma mater of <PERSON>([\.A-Za-z\s\']*)</PERSON>|to [\sa-zA-Z]*<PERSON>([\.A-Za-z\s\']*)</PERSON>'
		else: 
			pattern = '<PERSON>([\.A-Za-z\s\']*)</PERSON>'
	elif relation == "univ_president_of":
		if clue.count("ORGANIZATION")>2:
			pattern = "<ORGANIZATION>([&\.A-Za-z\s\']*)</ORGANIZATION> (is|was|has been) [\sa-zA-Z]*(headed|led|founded)|<ORGANIZATION>([&\.A-Za-z\s\']*)</ORGANIZATION>\'s [\sa-zA-Z]*(president|leader|head|founder)|(president|leader|head|founder)[\sA-Za-z<>/\']*of [\sA-Za-z<>/\']*<ORGANIZATION>([&\.A-Za-z\s\']*)</ORGANIZATION>|<ORGANIZATION>([&\.A-Za-z\s\']*)</ORGANIZATION> (president|leader|head|founder)|(headed|led|founded|in charge of) <ORGANIZATION>([&\.A-Za-z\s\']*)</ORGANIZATION>"
		else:
			pattern = "<ORGANIZATION>([\.A-Za-z\s\'&]*)</ORGANIZATION>"
	elif relation == "headquarters_loc":
		if clue.count("ORGANIZATION")>2:
	                pattern = "<ORGANIZATION>([&\.A-Za-z\s\']*)</ORGANIZATION> [\sa-zA-Z]*(based|headquartered|headquared|run|located|is in)|headquarters of <ORGANIZATION>([&\.A-Za-z\s\']*)</ORGANIZATION>|([&\.A-Za-z\s\']*) is run|([\.A-Za-z\s\'&]*) [\sa-zA-Z]* based|([&\.A-Za-z\s\']*) is in|([&\.A-Za-z\s\']*)'s [\sa-zA-Z]* office|([&\.A-Za-z\s\']*)'s [\sa-zA-Z]* headquarters"
		else:
                        pattern = "<ORGANIZATION>([\.A-Za-z\s\']*)</ORGANIZATION>|([&\.A-Za-z\s\']*) is run|([\.A-Za-z\s\'&]*) [\sa-zA-Z]* based|([&\.A-Za-z\s\']*) is in|([&\.A-Za-z\s\']*)'s [\sa-zA-Z]* office|([&\.A-Za-z\s\']*)'s [\sa-zA-Z]* headquarters"
	elif relation == "born_in":
		if clue.count("PERSON") > 2:
			pattern = "<PERSON>([\.A-Za-z\s\']*)</PERSON>[\sa-zA-Z,<>/]* born [\sa-zA-Z,]*in|birthplace [\sa-zA-Z,<>/]*<PERSON>([\.A-Za-z\s\']*)</PERSON>"
		else:
			pattern = "<PERSON>([\.A-Za-z\s\']*)</PERSON>" 
	elif relation == "parent_org_of":
		pattern = "(([A-Z][&\.A-Za-z\s\']*)+) (is|was)[\sa-zA-Z<>/1-9,]*offshoot|(([A-Z][&\.A-Za-z\s\']*)+) (is|was) [\sa-zA-Z<>/1-9,]*organization|(([A-Z][&\.A-Za-z\s\']*)+) (is|was) [\sa-zA-Z<>/1-9,]*orginazation|parent [\sa-zA-Z]*<ORGANIZATION>([&\.A-Za-z\s\']*)</ORGANIZATION>|<ORGANIZATION>([&\.A-Za-z\s\']*)</ORGANIZATION> (is|was) [\sa-zA-Z]*offshoot|<ORGANIZATION>([&\.A-Za-z\s\']*)</ORGANIZATION> (is|was) [\sa-zA-Z]*organization|parent \sa-zA-Z]*(([A-Z][&\.A-Za-z\s\']*)+)|ORGANIZATION>([&\.A-Za-z\s\']*)</ORGANIZATION> (is|was) [\sa-zA-Z]*orginazation|parent [\sa-zA-Z<>/]*(([A-Z][&\.A-Za-z\s\']*)+)"
	elif relation == "mayor_of":
		if clue.count("LOCATION") < 4:
			pattern = "<LOCATION>([A-Za-z\s]*)</LOCATION>, ([A-Za-z]*)'s mayor|mayor of <LOCATION>([A-Za-z\s]*)</LOCATION>, ([A-Za-z]*)|<LOCATION>([A-Za-z\s]*)</LOCATION>, ([A-Za-z]*)[\sa-zA-Z<>/1-9,]*is[\sa-zA-Z<>/1-9,]*(led|headed|run) by|mayor of ([A-Za-z]*), <LOCATION>([A-Za-z\s]*)</LOCATION>|([A-Za-z]*), <LOCATION>([A-Za-z\s]*)</LOCATION>[\sa-zA-Z<>/1-9,]*is[\sa-zA-Z<>/1-9,]*(led|headed|run) |mayor of ([A-Za-z]*), ([A-Za-z]*)"
		else:
			pattern = "<LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION> [\sa-zA-Z<>/1-9,]*is led by|mayor of <LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION>|<LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION>'s mayor|<LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION>[\sa-zA-Z<>/1-9,]* is [\sa-zA-Z<>/1-9,]*(headed|led|run)|<LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION>'s mayor"
	elif relation == "univ_in":
		pattern = "school [\sa-zA-Z<>/1-9,]*in <LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION>|college [\sa-zA-Z<>/1-9,]*in <LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION>|university [\sa-zA-Z<>/1-9,]*in <LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION>|<LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION> [\sa-zA-Z<>/1-9,]*school|<LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION> [\sa-zA-Z<>/1-9,]*college|<LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION> [\sa-zA-Z<>/1-9,]*university|<LOCATION>([A-Za-z\s]*)</LOCATION>, ([A-Za-z]*) is home|([A-Za-z]*), <LOCATION>([A-Za-z\s]*)</LOCATION> is home|<LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION> in|<LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION> is home|based in <LOCATION>([A-Za-z\s]*)</LOCATION>, <LOCATION>([A-Za-z\s]*)</LOCATION>|located in <LOCATION>([A-Za-z\s]*)</LOCATION>, ([A-Za-z]*)|in <LOCATION>([A-Za-z\s]*)</LOCATION>, ([A-Za-z]*)"
	elif relation == "year_of_birth":
		pattern = "<PERSON>([\.A-Za-z\s\']*)</PERSON>[\sa-zA-Z<>/1-9,]*(was|is) born|<PERSON>([\.A-Za-z\s\']*)</PERSON>'s birthday|birthday of [\sa-zA-Z1-9,]*<PERSON>([\.A-Za-z\s\']*)</PERSON>"
	elif relation == "year_of_death":
		pattern = "<PERSON>([\.A-Za-z\s\']*)</PERSON>[\sa-zA-Z<>/1-9,]*died|<PERSON>([\.A-Za-z\s\']*)</PERSON>[\sa-zA-Z<>/1-9,]*passed"
	if len(pattern) > 0:
		answer= self.getMatch(clue, pattern, relation)
		if answer == "no match" and relation != "mayor_of" and relation != "univ_in":
			stop_index = clue.find("</")
			start_index = clue.find(">")
			answer = clue[start_index+1:stop_index]
		if answer == "no match" and clue.count("</") > 1 and (relation == "mayor_of" or relation == "univ_in"):
			stop1 = clue.find("</")
			stop2 = clue.find("</", stop1+2)
			start1 = clue.find(">")
			temp = clue.find(">", start1+1)
			start2 = clue.find(">", temp+1)
			answer = clue[start1+1:stop1] + ", "+clue[start2+1:stop2]
		return answer;
	return 'Gene Autry'
	
    def parseClues(self, clues):
        """Parse each clue and return a list of parses, one for each clue."""
        parses = []
        for clue in clues:
            feature = self.feature_extractor(clue)
            clue_relation = self.classifer.classify(feature)
            clue_entity = self.getEntity(clue_relation, clue)
       #     if clue_entity == "no match":
        #        print(clue + " " + clue_relation)
            parses.append(clue_relation + ':' + clue_entity)
        return parses

    #### You should not need to change anything after this point. ####

    def evaluate(self, parsed_clues, gold_parsed_clues):
        """Shows how the ClueParser model will score on the training/development data."""
        correct_relations = 0
        correct_parses = 0
        for parsed_clue, gold_parsed_clue in it.izip(parsed_clues, gold_parsed_clues):
            split_parsed_clue = parsed_clue.split(":")
            split_gold_parsed_clue = gold_parsed_clue.split(":")
            if split_parsed_clue[0] == split_gold_parsed_clue[0]:
                correct_relations += 1
                if (split_parsed_clue[1] == split_gold_parsed_clue[1] or
                        split_parsed_clue[1] == "The " + split_gold_parsed_clue[1] or
                        split_parsed_clue[1] == "the " + split_gold_parsed_clue[1]):
                    correct_parses += 1
        print "Correct Relations: %d/%d" % (correct_relations, len(gold_parsed_clues))
        print "Correct Full Parses: %d/%d" % (correct_parses, len(gold_parsed_clues))
        print "Total Score: %d/%d" % (correct_relations + correct_parses, 2 * len(gold_parsed_clues))

def loadList(file_name):
    """Loads text files as lists of lines. Used in evaluation."""
    with open(file_name) as f:
        l = [line.strip() for line in f]
    return l

def main():
    """Tests the model on the command line. This won't be called in
        scoring, so if you change anything here it should only be code 
        that you use in testing the behavior of the model."""

    clues_file = "data/part1-clues.txt"
    parsed_clues_file = "data/part1-parsedclues.txt"
    cp = ClueParser()

    clues = loadList(clues_file)
    gold_parsed_clues = loadList(parsed_clues_file)
    assert(len(clues) == len(gold_parsed_clues))

    cp.train(clues, gold_parsed_clues)
    parsed_clues = cp.parseClues(clues)
    cp.evaluate(parsed_clues, gold_parsed_clues)
    if len(sys.argv) > 1 and sys.argv[1] == '-v':
        print "\nValidation results:"
        clues_file = "data/part2-clues-val.txt"
        parsed_clues_file = "data/part2-parses-val.txt"
        
        clues = loadList(clues_file)
        gold_parsed_clues = loadList(parsed_clues_file)
        assert(len(clues) == len(gold_parsed_clues))
        
        parsed_clues = cp.parseClues(clues)
        cp.evaluate(parsed_clues, gold_parsed_clues)

if __name__ == '__main__':
    main()
