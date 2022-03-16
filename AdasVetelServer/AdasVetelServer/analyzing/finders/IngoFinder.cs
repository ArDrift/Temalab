using AdasVetelServer.dbHandler;
using AdasVetelServer.model;
using System.Collections.Generic;

namespace AdasVetelServer.logic
{
    class IngoFinder : FinderBase<Ingo>
    {
        public static IngoFinder Instance { get; } = new IngoFinder();
        private IngoFinder() { }

        protected override void fillFindable()
        {
            ;
        }

        protected override void formatAttribute()
        {
            ;
        }
        protected override void validateFindable()
        {
            ;
        }
        protected override List<Ingo> getDatabaseElements()
        {
            return dbHandler.getIngosagok();
        }
        protected override void insertFindableToDatabase()
        {
            dbHandler.insertIngoToDatabase(findable);
        }
    }
}
