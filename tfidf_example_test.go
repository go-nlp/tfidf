package tfidf_test

import (
	"fmt"
	"strings"

	. "github.com/go-nlp/tfidf"
)

var mobydick = []string{
	"Call me Ishmael .",
	"Some years ago -- never mind how long precisely -- having little or no money in my purse , and nothing particular to interest me on shore , I thought I would sail about a little and see the watery part of the world .",
	"It is a way I have of driving off the spleen and regulating the circulation .",
	"Whenever I find myself growing grim about the mouth ; ",
	"whenever it is a damp , drizzly November in my soul ; ",
	"whenever I find myself involuntarily pausing before coffin warehouses , and bringing up the rear of every funeral I meet ; ",
	"and especially whenever my hypos get such an upper hand of me , that it requires a strong moral principle to prevent me from deliberately stepping into the street , and methodically knocking people's hats off -- then , I account it high time to get to sea as soon as I can .",
	"This is my substitute for pistol and ball . ",
	"With a philosophical flourish Cato throws himself upon his sword ; ",
	"I quietly take to the ship . There is nothing surprising in this .",
	"If they but knew it , almost all men in their degree , some time or other , cherish very nearly the same feelings towards the ocean with me .",
}

type doc []int

func (d doc) IDs() []int { return []int(d) }

func makeCorpus(a []string) (map[string]int, []string) {
	retVal := make(map[string]int)
	invRetVal := make([]string, 0)
	var id int
	for _, s := range a {
		for _, f := range strings.Fields(s) {
			f = strings.ToLower(f)
			if _, ok := retVal[f]; !ok {
				retVal[f] = id
				invRetVal = append(invRetVal, f)
				id++
			}
		}
	}
	return retVal, invRetVal
}

func makeDocuments(a []string, c map[string]int) []Document {
	retVal := make([]Document, 0, len(a))
	for _, s := range a {
		var ts []int
		for _, f := range strings.Fields(s) {
			f = strings.ToLower(f)
			id := c[f]
			ts = append(ts, id)
		}
		retVal = append(retVal, doc(ts))
	}
	return retVal
}

func Example() {
	corpus, invCorpus := makeCorpus(mobydick)
	docs := makeDocuments(mobydick, corpus)
	tf := New()

	for _, doc := range docs {
		tf.Add(doc)
	}
	tf.CalculateIDF()

	fmt.Println("IDF:")
	for i, w := range invCorpus {
		fmt.Printf("\t%q: %1.1f\n", w, tf.IDF[i])
		if i >= 10 {
			break
		}
	}

	scores := tf.Score(docs[0])
	fmt.Printf("Scores: %1.1f", scores)
	// Output:
	// IDF:
	// 	"call": 2.4
	// 	"me": 1.0
	// 	"ishmael": 2.4
	// 	".": 0.5
	// 	"some": 1.7
	// 	"years": 2.4
	// 	"ago": 2.4
	// 	"--": 1.7
	// 	"never": 2.4
	// 	"mind": 2.4
	// 	"how": 2.4
	// Scores: [1.2 1.3 1.2 0.9]
}
