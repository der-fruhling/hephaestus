HASM := hasm
HDIS := hdis
HLINK := hlink

TARGET := wasm32/component

all: linked.hob

clean:
	rm *.hob linked.heph -f

linked.heph: linked.hob
	${HDIS} -o $@ $<

linked.hob: test1.hob test2.hob
	${HLINK} -x ${TARGET} -o $@ $+

%.hob: %.heph
	${HASM} -x ${TARGET} -o $@ $<

.PHONY : all clean
