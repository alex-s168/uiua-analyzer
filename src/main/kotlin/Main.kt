package me.alex_s168.uiua

import blitz.collections.stringify
import blitz.io.Path
import blitz.io.read
import blitz.str.flattenToString

fun main() {
    val test = Path.of("/home/alex/uiua-analyzer/test.uasm")
        .getFile()
        .read()
        .stringify()
        .flattenToString()
    val assembly = Assembly.parse(test)
    assembly.functions.forEach { (k, v) ->
        println("function \"$k\" with ${v.signature}")
        val ast = astify(v.children)
        ast.printAst(2)
    }
}