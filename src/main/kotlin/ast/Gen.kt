package me.alex_s168.uiua.ast

import blitz.Either
import blitz.collections.addFront
import blitz.collections.removeLastInto
import me.alex_s168.uiua.*
import kotlin.math.max

fun astify(input: List<Instr>): ASTRoot {
    val tempStack = mutableMapOf<String, MutableList<AstNode>>()
    fun getTempStack(stack: String) =
        tempStack.getOrPut(stack) { mutableListOf() }

    val stack = mutableListOf<AstNode>()

    var argId = 0
    var argCount = 0

    val flagsAndComments: MutableList<Instr> = mutableListOf()

    fun <C: MutableList<AstNode>> copyLastInto(num: Int, dest: C): C {
        repeat(num - stack.size) {
            stack.addFront(AstNode(Either.ofB(Either.ofA(AstArgNode(argId++)))))
            argCount = max(argCount, argId)
        }
        dest += stack.takeLast(num)
        return dest
    }

    fun <C: MutableList<AstNode>> removeLastInto(num: Int, dest: C): C {
        repeat(num - stack.size) {
            stack.addFront(AstNode(Either.ofB(Either.ofA(AstArgNode(argId++)))))
            argCount = max(argCount, argId)
        }
        stack.removeLastInto(num, dest)
        return dest
    }

    input.forEachIndexed { _, op ->
        when (op) {
            is CopyTempStackInstr -> {
                val value = copyLastInto(op.count, mutableListOf())
                repeat(op.count - value.size) {
                    value += AstNode(Either.ofB(Either.ofA(AstArgNode(argId))))
                }
                argCount = argId + op.count - value.size
                getTempStack(op.stack).addAll(value)
            }

            is PushTempStackInstr -> {
                val value = removeLastInto(op.count, mutableListOf())
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

                val args = removeLastInto(sig.inputs, mutableListOf())
                if (sig.outputs > 0) {
                    val node = AstNode(Either.ofA(AstInstrNode(op, args)))

                    repeat(sig.outputs - 1) {
                        stack.add(AstNode(Either.ofB(Either.ofB(AstResExtendNode(node, it + 1)))))
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