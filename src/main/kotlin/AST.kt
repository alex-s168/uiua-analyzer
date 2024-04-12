package me.alex_s168.uiua

import blitz.Either

data class AstInstrNode(
    val instr: Instr?,
    val children: MutableList<AstNode>
)

data class AstArgNode(
    val id: Int
)

typealias AstNode = Either<AstInstrNode, AstArgNode>

private fun astifyLoop(input: Iterator<Instr>, weNeed: Int): MutableList<AstNode> {
    var weHave = 0
    val out = mutableListOf<AstNode>()
    while (weHave < weNeed && input.hasNext()) {
        val instr = input.next()
        val sig = signature(instr)

        val node = AstInstrNode(instr, astifyLoop(input, sig.inputs))
        out.add(Either.ofA(node))
        weHave += sig.outputs
    }
    repeat(weNeed - weHave) {
        out.add(Either.ofA(AstInstrNode(null, mutableListOf()))) // arg get
    }
    out.reverse()
    return out
}

private fun astifyFixArgs(nodes: MutableList<AstNode>, nextIn: Int): Int {
    var next = nextIn

    nodes.toList().forEachIndexed { index, node ->
        node.mapA {
            if (it.instr == null) {
                nodes[index] = Either.ofB(AstArgNode(next))
                next ++
            } else {
                next += astifyFixArgs(it.children, next)
            }
        }
    }

    return next
}

fun astify(input: List<Instr>): List<AstNode> {
    val prefix = input.reversed().asIterable().iterator()

    val out = mutableListOf<AstNode>()
    while (prefix.hasNext()) {
        val instr = prefix.next()
        val sig = signature(instr)

        val node = AstInstrNode(instr, astifyLoop(prefix, sig.inputs))
        out.add(Either.ofA(node))
    }
    out.reverse()
    astifyFixArgs(out, 0)
    return out
}

fun List<AstNode>.printAst(indent: Int = 0) {
    forEach { node ->
        print(" ".repeat(indent))
        node.mapB { it: AstArgNode ->
            println("arg ${it.id}")
        }.mapA {
            println("op ${it.instr}")
            it.children.printAst(indent + 2)
        }
    }
}