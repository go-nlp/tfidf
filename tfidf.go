// package tfidf is a lingo-friendly TF-IDF library
package tfidf

import (
	"math"
	"sync"

	"github.com/xtgo/set"
)

// TFIDF is a structure holding the relevant state information about TF/IDF
type TFIDF struct {
	// Term Frequency
	TF map[int]float64

	// Inverse Document Frequency
	IDF map[int]float64

	// docs is the count of documents
	docs int

	sync.Mutex
}

// Document is a representation of a document.
type Document interface {
	IDs() []int
}

// NewTFIDF creates a new TFIDF structure
func NewTFIDF() *TFIDF {
	return &TFIDF{
		TF:  make(map[int]float64),
		IDF: make(map[int]float64),
	}
}

// Add adds a document to the state
func (tf *TFIDF) Add(doc Document) {
	ids := doc.IDs()
	ints := make([]int, len(ids))
	copy(ints, ids)
	ints = set.Ints(ints)

	tf.Lock()
	for _, w := range ints {
		tf.TF[w]++
	}
	tf.Unlock()
	tf.docs++
}

// CalculateIDF calculates the inverse document frequency
func (tf *TFIDF) CalculateIDF() {
	docs := float64(tf.docs)
	tf.Lock()
	for t, f := range tf.TF {
		tf.IDF[t] = math.Log(docs / f)
	}
	tf.Unlock()
}

// Score calculates the TFIDF score (TF * IDF) for the document without adding the document to the tracked document count.
//
// This function is only useful for a handful of cases. It's recommended you write your own scoring functions.
func (tf *TFIDF) Score(doc Document) []float64 {
	ids := doc.IDs()

	retVal := make([]float64, len(ids))
	tf.Lock()
	for _, id := range ids {
		tf.TF[id]++
	}
	for i, id := range ids {
		retVal[i] = tf.TF[id]
	}
	for _, id := range ids {
		tf.TF[id]--
	}
	tf.Unlock()

	l := float64(len(ids))
	for i, freq := range retVal {
		retVal[i] = (freq / l) * tf.IDF[ids[i]]
	}
	return retVal
}
