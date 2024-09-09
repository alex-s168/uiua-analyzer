package me.alex_s168.uiua.ast

import blitz.Either
import blitz.collections.addFront
import blitz.collections.removeLast
import blitz.collections.removeLastInto
import me.alex_s168.uiua.*
import kotlin.math.max

fun astify(input: List<Instr>): ASTRoot {
    println(input)

    val tempStack = mutableMapOf<String, MutableList<AstNode>>()
    fun getTempStack(stack: String) =
        tempStack.getOrPut(stack) { mutableListOf() }

    val stack = mutableListOf<AstNode>()

    var argId = 0
    var argCount = 0

    val flagsAndComments: MutableList<Instr> = mutableListOf()

    input.forEachIndexed { _, op ->
        when (op) {
            is CopyTempStackInstr -> {
                val value = stack.takeLast(op.count).toMutableList()
                repeat(op.count - value.size) {
                    value += AstNode(Either.ofB(Either.ofA(AstArgNode(argId))))
                }
                argCount = argId + op.count - value.size
                getTempStack(op.stack).addAll(value)
            }

            is PushTempStackInstr -> {
                val value = stack.removeLastInto(op.count, mutableListOf())
                repeat(op.count - value.size) {
                    value += AstNode(Either.ofB(Either.ofA(AstArgNode(argId))))
                }
                argCount = argId + op.count - value.size
                getTempStack(op.stack).addAll(value)
            }

            is PopTempStackInstr -> {
                val value = getTempStack(op.stack).removeLastInto(op.count)
                stack.addAll(value)
            }

            else -> {
                val sig = signature(op) { stack[stack.size - 1 - it] }

                repeat(sig.inputs - stack.size) {
                    stack.addFront(AstNode(Either.ofB(Either.ofA(AstArgNode(argId++)))))
                    argCount = max(argCount, argId)
                }

                val args = stack.removeLastInto(sig.inputs)
                if (sig.outputs > 0) {
                    val node = AstNode(Either.ofA(AstInstrNode(op, args)))

                    repeat(sig.outputs - 1) {
                        stack.add(AstNode(Either.ofB(Either.ofB(AstResExtendNode(node)))))
                    }

                    stack.add(node)
                } else if (op is FlagInstr || op is CommentInstr) {
                    stack.lastOrNull()?.flagsAndComments?.add(op) ?: {
                        flagsAndComments.add(op)
                    }
                }
            }
        }
    }

    require(tempStack.values.all { it.isEmpty() })

    return ASTRoot(argCount, stack, flagsAndComments, null)
}