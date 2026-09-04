## ADDED Requirements

### Requirement: Dependabot watches hashed pip requirements
The repository SHALL declare Dependabot version updates for the pip ecosystem in `/src` so hashed `requirements.txt` pins can be bumped.

#### Scenario: dependabot.yml enables pip in src
- **WHEN** `.github/dependabot.yml` is read
- **THEN** it contains `package-ecosystem: pip` and `directory: /src`
