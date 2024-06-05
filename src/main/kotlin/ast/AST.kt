package me.alex_s168.uiua.ast

import blitz.Either
import me.alex_s168.uiua.*

data class AstInstrNode(
    val instr: Instr,
    val children: MutableList<AstNode>
)

data class AstArgNode(
    val id: Int
)

class AstResExtendNode

data class AstNode(
    val value: Either<AstInstrNode, Either<AstArgNode, AstResExtendNode>>,
    val flagsAndComments: MutableList<Instr> = mutableListOf(),
)

data class ASTRoot(
    val args: Int,
    val children: MutableList<AstNode>,
    val flagsAndComments: MutableList<Instr> = mutableListOf(),
)

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
            println("extend")
        }
    }
}

fun AstNode.printAst(indent: Int) =
    listOf(this).printAst(indent)