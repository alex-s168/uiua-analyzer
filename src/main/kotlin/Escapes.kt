package me.alex_s168.uiua

fun unescape(str: String): String {
    val out = StringBuilder()
    var escaped = false

    for (char in str) {
        if (escaped) {
            escaped = false
            val e = when (char) {
                'n' -> '\n'
                'r' -> '\r'
                't' -> '\t'
                '\\' -> '\\'
                '"' -> '"'
                '\'' -> '\''
                else -> error("Unexpected character '$char'")
            }
            out.append(e)
        } else if (char == '\\') {
            escaped = true
        } else {
            out.append(char)
        }
    }

    return out.toString()
}