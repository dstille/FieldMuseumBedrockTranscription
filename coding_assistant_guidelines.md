# Coding Assistant Guidelines

## Purpose
This document outlines guidelines and best practices for interacting with and utilizing coding assistants effectively.

## General Guidelines

- **Be specific in your requests**: Provide context, requirements, and expected behavior
- **Review generated code**: Always review and understand code before implementing
- **Iterative refinement**: Use follow-up questions to refine and improve generated solutions
- **Provide feedback**: Help the assistant learn your preferences and coding style

## Prompt Engineering Tips

- Start with clear problem statements
- Include relevant context (language, framework, constraints)
- Specify expected output format
- Break complex tasks into smaller steps
- Reference existing code patterns when applicable

## Code Quality Expectations

- Generated code should follow project style guidelines
- Include appropriate error handling
- Be security-conscious
- Include necessary comments and documentation
- Consider performance implications

## Example Prompts

### Good Example
```
Create a Python function that validates email addresses using regex. 
The function should:
- Accept a string parameter
- Return a boolean (true if valid)
- Handle common email formats
- Include docstrings and type hints
```

### Poor Example
```
Write email validation code
```

## Security Considerations

- Never share sensitive credentials or tokens with coding assistants
- Review generated code for security vulnerabilities
- Be cautious with generated code that:
  - Makes network requests
  - Accesses file systems
  - Executes shell commands
  - Handles user input without validation

## Effective Collaboration Workflow

1. Start with a clear problem definition
2. Request a solution approach before implementation
3. Review and refine the generated code
4. Test thoroughly before deployment
5. Document lessons learned for future interactions

---

*These guidelines are a living document and should be updated as best practices evolve.*