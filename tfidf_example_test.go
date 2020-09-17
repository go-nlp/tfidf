package tfidf_test

import (
	"fmt"
	"math"
	"sort"
	"strings"

	. "github.com/go-nlp/tfidf"
	"github.com/xtgo/set"
	"gorgonia.org/tensor"
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

type docScore struct {
	id    int
	score float64
}

type docScores []docScore

func (ds docScores) Len() int           { return len(ds) }
func (ds docScores) Less(i, j int) bool { return ds[i].score < ds[j].score }
func (ds docScores) Swap(i, j int) {
	ds[i].score, ds[j].score = ds[j].score, ds[i].score
	ds[i].id, ds[j].id = ds[j].id, ds[i].id
}

func cosineSimilarity(queryScore []float64, docIDs []int, relVec []float64) docScores {
	// special case
	if len(docIDs) == 1 {
		// even more special case!
		if len(queryScore) == 1 {
			return docScores{
				{docIDs[0], queryScore[0] * relVec[0]},
			}
		}

		q := tensor.New(tensor.WithBacking(queryScore))
		m := tensor.New(tensor.WithBacking(relVec))
		score, err := q.Inner(m)
		if err != nil {
			panic(err)
		}
		return docScores{
			{docIDs[0], score.(float64)},
		}
	}

	m := tensor.New(tensor.WithShape(len(docIDs), len(queryScore)), tensor.WithBacking(relVec))
	q := tensor.New(tensor.WithShape(len(queryScore)), tensor.WithBacking(queryScore))
	dp, err := m.MatVecMul(q)
	if err != nil {
		panic(err)
	}

	m2, err := tensor.Square(m)
	if err != nil {
		panic(err)
	}

	normDocs, err := tensor.Sum(m2, 1)
	if err != nil {
		panic(err)
	}

	normDocs, err = tensor.Sqrt(normDocs)
	if err != nil {
		panic(err)
	}

	q2, err := tensor.Square(q)
	if err != nil {
		panic(err)
	}
	normQt, err := tensor.Sum(q2)
	if err != nil {
		panic(err)
	}
	normQ := normQt.Data().(float64)
	normQ = math.Sqrt(normQ)

	norms, err := tensor.Mul(normDocs, normQ)
	if err != nil {
		panic(err)
	}

	cosineSim, err := tensor.Div(dp, norms)
	if err != nil {
		panic(err)
	}
	csData := cosineSim.Data().([]float64)

	var ds docScores
	for i, id := range docIDs {
		score := csData[i]
		ds = append(ds, docScore{id: id, score: score})
	}
	return ds

}

func contains(query Document, in []Document, tf *TFIDF) (docIDs []int, relVec []float64) {
	q := query.IDs()
	q = set.Ints(q) // unique words only
	for i := range in {
		doc := in[i].IDs()

		var count int
		var relevant []float64
		for _, wq := range q {
		inner:
			for _, wd := range doc {
				if wq == wd {
					count++
					break inner
				}
			}
		}
		if count == len(q) {
			// calculate the score of the doc
			score := tf.Score(in[i])
			// get the  scores of the relevant words
			for _, wq := range q {
			inner2:
				for j, wd := range doc {
					if wd == wq {
						relevant = append(relevant, score[j])
						break inner2
					}
				}
			}
			docIDs = append(docIDs, i)
			relVec = append(relVec, relevant...)
		}
	}
	return
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

	// now we search

	// "ishmael" is a query
	ishmael := doc{corpus["ishmael"]}

	// "whenever i find" is another query
	whenever := doc{corpus["whenever"], corpus["i"], corpus["find"]}

	// step1: score the queries
	ishmaelScore := tf.Score(ishmael)
	wheneverScore := tf.Score(whenever)

	// step2: find the docs that contains the queries.
	// if there are no docs, oops.
	ishmaelDocs, ishmaelRelVec := contains(ishmael, docs, tf)
	wheneverDocs, wheneverRelVec := contains(whenever, docs, tf)

	// step3: calculate the cosine similarity
	ishmaelRes := cosineSimilarity(ishmaelScore, ishmaelDocs, ishmaelRelVec)
	wheneverRes := cosineSimilarity(wheneverScore, wheneverDocs, wheneverRelVec)

	// step4: sort the results
	sort.Sort(sort.Reverse(ishmaelRes))
	sort.Sort(sort.Reverse(wheneverRes))

	fmt.Printf("Relevant Docs to \"Ishmael\":\n")
	for _, d := range ishmaelRes {
		fmt.Printf("\tID   : %d\n\tScore: %1.3f\n\tDoc  : %q\n", d.id, d.score, mobydick[d.id])
	}
	fmt.Println("")
	fmt.Printf("Relevant Docs to \"whenever i find\":\n")
	for _, d := range wheneverRes {
		fmt.Printf("\tID   : %d\n\tScore: %1.3f\n\tDoc  : %q\n", d.id, d.score, mobydick[d.id])
	}
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
	// Relevant Docs to "Ishmael":
	//	ID   : 0
	//	Score: 1.437
	//	Doc  : "Call me Ishmael ."
	//
	// Relevant Docs to "whenever i find":
	//	ID   : 5
	//	Score: 0.985
	//	Doc  : "whenever I find myself involuntarily pausing before coffin warehouses , and bringing up the rear of every funeral I meet ; "
	//	ID   : 3
	//	Score: 0.962
	//	Doc  : "Whenever I find myself growing grim about the mouth ; "

}
