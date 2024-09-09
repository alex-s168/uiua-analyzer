package me.alex_s168.uiua.ast

import blitz.Either
import blitz.flatten
import blitz.mapBA
import blitz.mapBB
import me.alex_s168.uiua.*

data class AstInstrNode(
    val instr: Instr,
    val children: MutableList<AstNode>
)

data class AstArgNode(
    val id: Int
)

data class AstResExtendNode(
    val of: AstNode
)

class AstNode(
    val value: Either<AstInstrNode, Either<AstArgNode, AstResExtendNode>>,
    val flagsAndComments: MutableList<Instr> = mutableListOf(),
)

class ASTRoot(
    val args: Int,
    val children: MutableList<AstNode>,
    val flagsAndComments: MutableList<Instr> = mutableListOf(),
)

fun AstNode.flatten(): List<AstNode> {
    return if (value.isA) {
        value.getA().children.flatMap { it.flatten() } + this
    } else {
        val b = value.getB()
        if (b.isA) {
            listOf(this)
        }
        else {
            b.getB().of.flatten() + this
        }
    }
}

fun List<AstNode>.printAst(indent: Int = 0) {
    forEach { node ->
        node.flagsAndComments.forEach {
            print(" ".repeat(indent))
            println("@$it")
        }
        print(" ".repeat(indent))
        node.value.mapA {
            println("op ${it.instr}")
            it.children.printAst(indent + 2)
        }.mapBA {
            println("arg ${it.id}")
        }.mapBB {
            println("extend ${it.of.value.mapBA { "arg ${it.id}" }.mapBB { "extend" }.mapA { it.instr }.flatten()}")
        }
    }
}

fun AstNode.printAst(indent: Int) =
    listOf(this).printAst(indent)