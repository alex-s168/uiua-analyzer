package me.alex_s168.uiua.ast

import blitz.*
import me.alex_s168.uiua.*

class AstInstrNode(
    val instr: Instr,
    val children: MutableList<AstNode>
)

class AstArgNode(
    val id: Int
)

class AstResExtendNode(
    val of: AstNode,
    val resIdx: Int,
)

class AstNode(
    val value: Either<AstInstrNode, Either<AstArgNode, AstResExtendNode>>,
    val flagsAndComments: MutableList<Instr> = mutableListOf(),
)

class ASTRoot(
    val args: Int,
    val children: MutableList<AstNode>,
    val flagsAndComments: MutableList<Instr> = mutableListOf(),
    var functionName: String?,
)

fun AstNode.flatten(): List<AstNode> =
    value.mapA {
        it.children.flatMap { it.flatten() } + this
    }.mapBA {
        listOf(this)
    }.mapBB {
        it.of.flatten() + this
    }.partiallyFlattenB().flatten()

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