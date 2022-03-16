using AdasVetelServer.dbHandler;
using AdasVetelServer.logic.nep;
using AdasVetelServer.model;
using System.Collections.Generic;

namespace AdasVetelServer.logic
{
    abstract class FinderBase<T> where T : DbElement, ICheckable<T>, new()
    {
        protected DbHandler dbHandler = null;
        protected List<T> productionHistory = new List<T>();
        public List<T> ProductionHistory
        {
            get
            {
                return productionHistory;
            }
        }

        protected Dictionary<string, List<MatchResult>> matchesDict;
        protected int start, end;
        protected T findable;
        protected int validValues;
        public T findOne(Dictionary<string, List<MatchResult>> matchesDict, int start = int.MinValue, int end = int.MaxValue)
        {
            while (productionHistory.Count > 30) {
                productionHistory.RemoveAt(0);
            }

            dbHandler = TextAnalyzer.Instance.dbh;
            validValues = 0;
            this.matchesDict = matchesDict;
            this.start = start;
            this.end = end;
            findable = new T();


            fillFindable();
            validateFindable();
            formatAttribute();

            if (!findable.isValid)
            {
                return null;
            }    

            T element = productionHistory.Find(e => e.valueEquals(findable));
            if (element != null) {
                return element;
            }
            
            CheckAndInsertToDatabase();
            productionHistory.Add(findable);
            return findable;

        }
        private bool CheckAndInsertToDatabase()
        {           
            List<T> elements = getDatabaseElements();
            T element = elements.Find(e => e.valueEquals(findable));
            if (element == null)
            {
                insertFindableToDatabase();
                return true;
            }
            findable = element;
            return false;
        }
        protected abstract void fillFindable();
        protected abstract void formatAttribute();
        protected abstract void validateFindable();
        protected abstract List<T> getDatabaseElements();
        protected abstract void insertFindableToDatabase();

    }
}
