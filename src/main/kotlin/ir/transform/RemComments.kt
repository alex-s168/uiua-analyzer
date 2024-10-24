package me.alex_s168.uiua.ir.transform

import me.alex_s168.uiua.CommentInstr
import me.alex_s168.uiua.ir.optAwayPass
import me.alex_s168.uiua.ir.parallelWithoutDeepCopy

val remComments = optAwayPass(
    "rm comments",
    { it.instr is CommentInstr },
    { true }
).parallelWithoutDeepCopy()