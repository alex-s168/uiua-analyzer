package me.alex_s168.uiua.ir

import me.alex_s168.uiua.log

val verifyBlock = Pass<Unit>("verify block") { block, _ ->
    block.instrs.forEachIndexed { i, it ->
        val correct = it
            .deepCopy()
        runCatching {
            correct.inferTypes(block, {}, block.fillArg?.type, verify = true)
        }.onFailure {
            log("in instruction $i:")
            throw it
        }

        if (!it.similar(correct)) {
            log("in instruction $i:\n  expected: $correct\n  got: $it")
            error("instruction has invalid types")
        }
    }
}.parallelWithoutDeepCopy()