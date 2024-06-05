package me.alex_s168.uiua.ast

import blitz.Either
import blitz.collections.removeLastInto
import me.alex_s168.uiua.*

fun astify(input: List<Instr>): ASTRoot {
    val stack = mutableListOf<AstNode>()

    var argId = 0

    val flagsAndComments: MutableList<Instr> = mutableListOf()

    input.forEach { op ->
        val sig = signature(op)

        if (sig.inputs > stack.size) {
            stack.addFront(AstNode(Either.ofB(Either.ofA(AstArgNode(argId ++)))))
        }

        val args = stack.removeLastInto(sig.inputs)
        if (sig.outputs > 0) {
            stack.add(AstNode(Either.ofA(AstInstrNode(op, args))))

            repeat(sig.outputs - 1) {
                stack.add(AstNode(Either.ofB(Either.ofB(AstResExtendNode()))))
            }
        } else if (op is FlagInstr || op is CommentInstr) {
            stack.lastOrNull()?.flagsAndComments?.add(op) ?: {
                flagsAndComments.add(op)
            }
        }
    }

    return ASTRoot(argId, stack, flagsAndComments)
}