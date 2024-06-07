package me.alex_s168.uiua.ast

import blitz.Either
import blitz.collections.addFront
import blitz.collections.removeLastInto
import me.alex_s168.uiua.*

fun astify(input: List<Instr>): ASTRoot {
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
                val value = stack.lastOrNull() ?: AstNode(Either.ofB(Either.ofA(AstArgNode(argId))))
                argCount = argId + 1
                getTempStack(op.stack).add(value)
            }

            is PopTempStackInstr -> {
                val value = getTempStack(op.stack).removeLast()
                stack.add(value)
            }

            else -> {
                val sig = signature(op) { stack[stack.size - 1 - it] }

                repeat(sig.inputs - stack.size) {
                    stack.addFront(AstNode(Either.ofB(Either.ofA(AstArgNode(argId++)))))
                    argCount = argId
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

    return ASTRoot(argCount, stack, flagsAndComments)
}